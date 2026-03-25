import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, R_Discriminator
from onpolicy.utils.util import update_linear_schedule
from torch.distributions import Categorical
import torch.nn.functional as F
import math

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, num_skills, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.num_skills = num_skills

        self.actor = R_Actor(args, self.num_skills, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.num_skills, self.share_obs_space, self.device)
        self.discriminator = R_Discriminator(args, self.share_obs_space, self.num_skills, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr,
                                            eps=self.opti_eps,
                                            weight_decay=self.weight_decay)
        

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, z_onehot, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 z_onehot,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, z_onehot, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, z_onehot, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, z_onehot, rnn_states_critic, masks)
        return values

    
    def compute_diayn_reward(self, cent_obs, rnn_states_discriminator, masks, true_skill):
        # 1. Ép kiểu từ Numpy sang Torch Tensor và đưa lên thiết bị (GPU/CPU)
        cent_obs = torch.from_numpy(cent_obs).float().to(self.device)
        rnn_states_discriminator = torch.from_numpy(rnn_states_discriminator).float().to(self.device)
        masks = torch.from_numpy(masks).float().to(self.device)
        true_skill = torch.from_numpy(true_skill).float().to(self.device)

        # 2. Forward qua Discriminator
        logits, rnn_states_discriminator = self.discriminator(cent_obs, rnn_states_discriminator, masks)
        
        # 3. Tính reward
        if len(true_skill.shape) == 3:
            true_skill_team = true_skill[:, 0, :] 
        else:
            true_skill_team = true_skill

        true_skills_idx = torch.argmax(true_skill_team, dim=-1)
        
        num_skills = logits.shape[-1]
        log_probs = F.log_softmax(logits, dim=-1)
        
        true_skills_idx = true_skills_idx.unsqueeze(-1) 
        skill_log_prob = log_probs.gather(dim=-1, index=true_skills_idx).squeeze(-1) 
        
        log_prior = math.log(1.0 / num_skills)
        intrinsic_reward = skill_log_prob - log_prior
            
        # 4. Trả về Numpy Array cho Runner xử lý tiếp
        return intrinsic_reward.unsqueeze(-1), rnn_states_discriminator
    
    def evaluate_actions(self, cent_obs, obs, z_onehot, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     z_onehot,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, z_onehot, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, z_onehot, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, z_onehot, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
