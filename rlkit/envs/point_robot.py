import gym
import numpy as np
from gym import spaces
from gym import Env
from gym.utils import seeding
from . import register_env

import pickle
import os
    

@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=True, n_tasks=2, seed=0):

        self.seed(seed)
        self.tasks = self.sample_tasks(n_tasks)
        self.goals = [self.tasks[i]['goal'] for i in range(len(self.tasks))]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()
        
    def sample_tasks(self, n_tasks):
        goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] 
                 for _ in range(n_tasks)]
        tasks = []
        for i in range(n_tasks):
            tasks.append({'goal':goals[i]})
        return tasks

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)
    
    def save_all_tasks(self, save_dir):
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)    


@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius
     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=True, n_tasks=2, goal_radius=0.2, seed=0):
        super().__init__(randomize_tasks, n_tasks)
        
        self.goal_radius = goal_radius
        
        self.tasks = self.sample_tasks(n_tasks)
        self.goals = [self.tasks[i]['goal'] for i in range(len(self.tasks))]

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r
    
    def sample_tasks(self, n_tasks):
        radius = 1.0
        angles = np.linspace(0, np.pi, num=n_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        np.random.shuffle(goals)
        goals = goals.tolist()
        
        tasks = []
        for i in range(n_tasks):
            tasks.append({'goal':goals[i]})
        return tasks
    
    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
    
    def save_all_tasks(self, save_dir):
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)    



@register_env('mixture-point-robot')
class Mixture_SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius
     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=True, n_tasks=2, goal_radius=0.2, seed=0):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius
        
        self.tasks = self.sample_tasks(n_tasks)
        self.goals = [self.tasks[i]['goal'] for i in range(len(self.tasks))]

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r
    
    def sample_tasks(self, n_tasks):
        radius = 1.0
        # goals sampled from discontinuous intervals with probability p
        angles = np.array([np.random.choice([np.random.uniform(0, (1.0/6.0)*np.pi),
                                             np.random.uniform((5.0/12.0)*np.pi, (7.0/12.0)*np.pi),
                                             np.random.uniform((5.0/6.0)*np.pi, np.pi)], 
                                            p=[1.0/3.0,1.0/3.0,1.0/3.0]) for i in range(n_tasks)])
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        np.random.shuffle(goals)
        goals = goals.tolist()
        
        tasks = []
        for i in range(n_tasks):
            tasks.append({'goal':goals[i]})
        return tasks
    
    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
    
    def save_all_tasks(self, save_dir):
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)    