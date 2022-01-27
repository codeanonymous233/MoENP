import numpy as np

from .ant import AntEnv
from . import register_env

import pickle
import os

@register_env('ant-multi-goal')
class AntMultiGoalEnv(AntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, max_eps=500, seed=0):
        self._max_eps = max_eps
        self._num_steps = 0
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super(AntMultiGoalEnv, self).__init__()
        self.seed(seed)
        
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        infos = dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )
        state = self.state_vector()
        done = False
        observation = self._get_obs()
        
        self._num_steps += 1
        if self._num_steps >= self._max_eps:
            done = True        
        
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        angles = np.array([np.random.choice([np.random.uniform(0, (1.0/6.0) * 2 * np.pi),
                                             np.random.uniform((5.0/12.0) * 2 * np.pi, (7.0/12.0) * 2 * np.pi),
                                             np.random.uniform((5.0/6.0) * 2 * np.pi,  2 * np.pi)], 
                                            p=[1.0/3.0,1.0/3.0,1.0/3.0]) for i in range(num_tasks)])
        
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(angles), r * np.sin(angles)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks
    
    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._num_steps = 0
        self._goal = self._task['goal']
        self.reset()
        
    def save_all_tasks(self, save_dir):
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)    
    
    