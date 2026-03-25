import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MATERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MATERunner, self).__init__(config)
        print("self.num_skills = ", self.num_skills)
        print("self.diayn_alpha = ", self.diayn_alpha)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # Sample skill riêng cho TỪNG thread và cố định suốt episode đó
            skill_ids = np.random.randint(0, self.num_skills, size=(self.n_rollout_threads,))
            #print("skill_ids = ", skill_ids)
            #time.sleep(10)
            z_onehot = np.zeros((self.n_rollout_threads, self.num_agents, self.num_skills), dtype=np.float32)
            for i in range(self.n_rollout_threads):
                z_onehot[i, :, skill_ids[i]] = 1.0
            
            # Gán skill vào bước 0 của buffer làm vạch xuất phát
            self.buffer.z_onehot[0] = z_onehot.copy()

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_discriminator_states, actions_env, diayn_reward = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                diayn_rewards = diayn_reward.reshape(self.n_rollout_threads, self.num_agents, 1)                
                # Cộng extrinsic + intrinsic reward
                rewards = rewards.reshape(self.n_rollout_threads, self.num_agents, 1)
                rewards = rewards + self.diayn_alpha * diayn_rewards
                
                # Đóng gói data truyền cho hàm insert (thêm discriminator_states và z_onehot)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_discriminator_states, z_onehot

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MATE":
                    env_infos = {}
                    
                    # 1. Khởi tạo danh sách để chứa dữ liệu từ 16 threads
                    all_thread_coverages = []
                    
                    for info in infos:
                        # Lấy coverage_rate từ dict của mỗi thread
                        if isinstance(info, dict) and 'coverage_rate' in info:
                            all_thread_coverages.append(info['coverage_rate'])
                            
                    # 2. Đẩy vào env_infos để log ra trung bình của 16 threads
                    if len(all_thread_coverages) > 0:
                        env_infos['coverage_rate'] = all_thread_coverages

                    # 3. Log thêm Individual Reward (nếu cần)
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            # MATE thường đặt tên key theo dạng 'agent0/individual_reward'
                            key = f'agent{agent_id}/individual_reward'
                            if isinstance(info, dict) and key in info:
                                idv_rews.append(info[key])
                        
                        if len(idv_rews) > 0:
                            env_infos[f'agent{agent_id}/individual_rewards'] = idv_rews

                # Các dòng dưới này phải thẳng hàng với khối "if self.env_name == 'MATE':"
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = self.envs.get_state()

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.z_onehot[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        
        # Lấy thêm discriminator outputs
        diayn_reward, rnn_discriminator_states = self.trainer.policy.compute_diayn_reward(
                                np.concatenate(self.buffer.share_obs[step]),
                                np.concatenate(self.buffer.rnn_discriminator_states[step]),
                                np.concatenate(self.buffer.masks[step]),
                                np.concatenate(self.buffer.z_onehot[step]),
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # Format discriminator return
        rnn_discriminator_states = np.array(np.split(_t2n(rnn_discriminator_states), self.n_rollout_threads))
        diayn_reward = _t2n(diayn_reward)

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(actions, axis=-1)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_discriminator_states, actions_env, diayn_reward

    def insert(self, data):
        # Bắt đủ 11 biến từ data package
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_discriminator_states, z_onehot = data

        # Xóa rnn_states nếu agent dead (dones == True)
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        rnn_discriminator_states[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_discriminator_states.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = self.envs.get_state()
        # Truyền đúng thứ tự chữ ký hàm trong SharedReplayBuffer.insert(...)
        self.buffer.insert(
            share_obs=share_obs, 
            obs=obs, 
            rnn_states_actor=rnn_states, 
            rnn_states_critic=rnn_states_critic, 
            rnn_discriminator_states=rnn_discriminator_states, 
            z_onehot=z_onehot, 
            actions=actions, 
            action_log_probs=action_log_probs, 
            value_preds=values, 
            rewards=rewards, 
            masks=masks
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # Tạo skill cho luồng Evaluate
        eval_skill_ids = np.random.randint(0, self.num_skills, size=(self.n_eval_rollout_threads,))
        eval_z_onehot = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_skills), dtype=np.float32)
        for i in range(self.n_eval_rollout_threads):
            eval_z_onehot[i, :, eval_skill_ids[i]] = 1.0

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                                                np.concatenate(eval_obs),
                                                np.concatenate(eval_z_onehot),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(eval_actions, axis=-1)
            else:
                raise NotImplementedError

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            # Tạo skill cho luồng Render
            render_skill_id = episode % self.num_skills 
            render_z_onehot = np.zeros((self.n_rollout_threads, self.num_agents, self.num_skills), dtype=np.float32)
            render_z_onehot[:, :, render_skill_id] = 1.0

            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                                                    np.concatenate(obs),
                                                    np.concatenate(render_z_onehot),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)