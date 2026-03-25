import sys
import numpy as np
import mate
from mate.agents import GreedyTargetAgent
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import time
import copy

def normalize_obs_optimized(obs):
    state = np.array(obs, copy=True, dtype=np.float32)
    
    n_c = int(state[0, 0])
    if n_c == 0:
        return state
        
    n_t = int(state[0, 1])
    n_o = int(state[0, 2])
    
    vals = state[:n_c, 3].astype(int)
    mapping_arr = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]], dtype=np.float32)
    encoded = np.zeros((n_c, 4), dtype=np.float32)
    valid_mask = (vals >= 0) & (vals <= 3)
    encoded[valid_mask] = mapping_arr[vals[valid_mask]]
    state[:n_c, 0:4] = encoded
    
    orig_rad = state[:n_c, 19].copy()
    
    state[:n_c, 4:16] /= 1000.0
    state[:n_c, 16:18] /= orig_rad[:, None] # Broadcasting chia cho radius của từng camera
    state[:n_c, 18] /= 180.0
    state[:n_c, 19] = 1.0
    state[:n_c, 20:22] /= 180.0
    
    if n_t > 0:
        t_start = 22
        t_end = t_start + 5 * n_t
        targets_view = state[:n_c, t_start:t_end].reshape(n_c, n_t, 5)
        targets_view[:, :, :3] /= 1000.0
        
    if n_o > 0:
        o_start = 22 + 5 * n_t
        o_end = o_start + 4 * n_o
        obs_view = state[:n_c, o_start:o_end].reshape(n_c, n_o, 4)
        obs_view[:, :, :3] /= 1000.0
        
    if n_c > 0:
        tm_start = 22 + 5 * n_t + 4 * n_o
        tm_end = tm_start + 7 * n_c
        tm_view = state[:n_c, tm_start:tm_end].reshape(n_c, n_c, 7)
        tm_view[:, :, :3] /= 1000.0
        tm_view[:, :, 3:5] /= orig_rad[:, None, None]
        tm_view[:, :, 5] /= 180.0
        
    return state

def normalize_state_optimized(s, num_cameras, num_targets, num_obstacles):
    state = np.array(s, copy=True, dtype=np.float32)
    if num_cameras > 0:
        cam_view = state[:9 * num_cameras].reshape(num_cameras, 9)
        cam_view[:, 0:3] /= 1000.0
        orig_rad = cam_view[:, 6].copy()
        cam_view[:, 3:5] /= orig_rad[:, None]
        cam_view[:, 5] /= 180.0
        cam_view[:, 6] = 1.0
        cam_view[:, 7:9] /= 180.0

    if num_targets > 0:
        t_start = 9 * num_cameras
        t_end = t_start + 14 * num_targets
        tgt_view = state[t_start:t_end].reshape(num_targets, 14)
        tgt_view[:, 0:3] /= 1000.0
        tgt_view[:, 4] /= 1000.0

    if num_obstacles > 0:
        o_start = 9 * num_cameras + 14 * num_targets
        o_end = o_start + 3 * num_obstacles
        obs_view = state[o_start:o_end].reshape(num_obstacles, 3)
        obs_view[:, :] /= 1000.0
        
    return state

class MATEEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        levels = config.levels
        base_env = mate.make("MultiAgentTracking-v0")
        base_env = mate.DiscreteCamera(base_env, levels=levels)
        base_env = mate.MultiCamera(
            base_env,
            target_agent=GreedyTargetAgent()
        )
        self.env = base_env
        self.n_agents = self.env.num_cameras
        self.n_targets = self.env.num_opponents
        self.n_obstacles = 9

        self.n_actions = self.env.action_space[0].n

        self.episode_limit = 900

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.last_obs = obs
        self.last_reward = 0.0
        flat_obs = [o.flatten() for o in obs]
        self.obs_shape = max(o.shape[0] for o in flat_obs)
        #print("obs_shape =", self.obs_shape)
        self.state_shape = self.get_state()[0].shape[0]
        self.observation_space = [
            spaces.Box(low=-1.0, high=+1.0, shape=(self.obs_shape,), dtype=np.float32) 
            for _ in range(self.n_agents)
        ]
        
        # 2. Share observation space (Global State): Cũng là LIST chứa Box cho từng agent
        self.share_observation_space = [
            spaces.Box(low=-1.0, high=+1.0, shape=(self.state_shape,), dtype=np.float32) 
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            spaces.Discrete(self.n_actions) 
            for _ in range(self.n_agents)
        ]

        self.t = 0

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.last_obs = obs
        self.t = 0
        self.last_reward = 0.0   
        return normalize_obs_optimized(obs=self.last_obs)

    def step(self, actions):
        reward_n = []
        done_n = []
        #print("actions = ", actions)
        #print("actions.shape = ", actions.shape)
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()

        actions = np.array(actions).astype(int)
        obs, reward, terminated, info = self.env.step(actions)
        reward = info[0]['coverage_rate']
        if isinstance(obs, tuple):
            obs = obs[0]

        reward = float(reward)
        self.last_obs = obs
        self.t += 1
        done = bool(terminated)
        if self.t >= self.episode_limit:
            done = True

        if not isinstance(info, dict):
            info = {}

        info["episode_limit"] = self.t >= self.episode_limit   
        self.last_reward = reward
        # ===== MULTI-AGENT FORMAT =====
        reward_n = [reward for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]
        return self.get_obs(), reward_n, done_n, info

    def get_obs(self):
        obs = normalize_obs_optimized(self.last_obs)
        return np.array(obs, dtype=np.float32)


    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.obs_shape

    def get_state(self):
        obs = self.last_obs
        camera_state = []
        for o in obs:
            camera_state.append(o[13:22])
        camera_state = np.array(camera_state).flatten()
        target_state = self.env.get_real_opponent_info()
        target_state = np.array(target_state).flatten()
        obstacle_state = self.env.get_obstacle_state()
        state = np.concatenate(
            [camera_state, target_state, obstacle_state],
            axis=0
        )
        normalized_state = normalize_state_optimized(s=state, num_cameras=self.n_agents, num_targets=self.n_targets, num_obstacles=self.n_obstacles)
        single_state = normalized_state.astype(np.float32)
        state_n = [single_state for _ in range(self.n_agents)]
        return state_n

    def get_state_size(self):
        return self.state_shape

    def get_avail_actions(self):

        return np.ones(
            (self.n_agents, self.n_actions),
            dtype=np.int32
        )

    def get_avail_agent_actions(self, agent_id):

        return np.ones(self.n_actions, dtype=np.int32)

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        return {}

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()

if __name__ == "__main__":

    print("===== TEST MATE ENV =====")
    config = {
        'levels': 9
    }
    env = MATEEnv(config=config)

    env_info = env.get_env_info()
    print("\nENV INFO:", env_info)

    obs = env.reset()

    print("\nReset done")
    print("obs shape:", obs.shape)

    state = env.get_state()

    print("state shape:", state.shape)

    print("\nRunning random policy...\n")

    done = False
    step = 0

    while not done:

        avail_actions = env.get_avail_actions()

        actions = []

        for agent_id in range(env.n_agents):

            action = np.random.randint(env.n_actions)

            actions.append(action)

        next_obs, reward, done, info = env.step(actions)

        obs = env.get_obs()
        state = env.get_state()
        print("obs = ", obs)
        print("state = ", state)
        print("------------------------------------------------")
        print(
            f"step {step:03d} | reward {reward:.4f} | "
            f"obs {obs.shape} | state {state.shape}"
        )

        step += 1

    print("\nEpisode finished")

    env.close()
