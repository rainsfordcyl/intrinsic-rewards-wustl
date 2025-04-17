import math
import numpy as np
import torch
from gymnasium.core import Wrapper

class SimHash:
    def __init__(self, state_dim, hash_dim):
        self.A = np.random.normal(0, 1, (hash_dim, state_dim))
        
    def __call__(self, state):
        return str(np.sign(self.A @ state.flatten()).tolist())

class CountBasedBonusWrapper(Wrapper):
    def __init__(self, env, beta=0.001):
        super().__init__(env)
        self.beta = beta
        self.counts = {}
        self.hash_fn = SimHash(147, 56)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Generate hash key
        hash_key = self.hash_fn(obs)
        
        # Update counts
        self.counts[hash_key] = self.counts.get(hash_key, 0) + 1
        
        # Calculate bonus
        bonus = self.beta / math.sqrt(self.counts[hash_key])
        reward += bonus
        
        return obs, reward, terminated, truncated, info