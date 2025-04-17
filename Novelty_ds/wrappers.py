import math
import random
import numpy as np
from collections import deque
import torch
from torch import nn, optim
from gymnasium.core import Wrapper
from models import SimHash, R_Model
from config import device

class CountBasedBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.M = {}
        self.hash = SimHash(147, 56)
        self.beta = 0.001

    def _update_count_dict(self, hash_key):
        pre_count = self.M.get(hash_key, 0)
        self.M[hash_key] = pre_count + 1

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_flatten = obs.flatten()
        hash_key = self.hash.hash(obs_flatten)
        self._update_count_dict(hash_key)
        bonus = self.beta / math.sqrt(self.M[hash_key])
        reward += bonus
        return obs, reward, terminated, truncated, info

class EpisodicCuriosityBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.M = []
        self.eps = []
        self.max_length = 10
        self.step_retrained_model = 0
        self.r_model = R_Model().to(device)
        self.optimizer = optim.Adam(self.r_model.parameters(), lr=1e-4)
        self.history = deque(maxlen=10)
        self.k = 5
        self.gamma = 1.2
        self.alpha = 0.001
        self.beta = 1
        self.model_trained = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.eps = [obs]
        self.M = [obs]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.eps.append(obs)
        if terminated or truncated:
            self.history.append(self.eps.copy())
            self.eps = []
            self.M = []
        self.step_retrained_model += 1
        if self.step_retrained_model == 30000 and self.history:
            X, y = self._create_training_data()
            self._train_r_model(X, y)
            self.step_retrained_model = 0
        if len(self.M) >= 2 and self.model_trained:
            bonus = self.alpha * (self.beta - self.r_model.get_reward(obs, self.M))
            reward += bonus
        if len(self.M) >= self.max_length:
            self.M.pop(random.randint(0, len(self.M)-1))
        self.M.append(obs)
        return obs, reward, terminated, truncated, info

    def _create_training_data(self):
        X, y = [], []
        for episode in self.history:
            for _ in range(30):
                idx = random.randint(0, len(episode)-2)
                max_step = min(self.k, len(episode)-idx-1)
                if max_step > 0:
                    step = random.randint(1, max_step)
                    X.append((episode[idx], episode[idx+step]))
                    y.append(1)
                    if idx + self.k*self.gamma < len(episode):
                        step = random.randint(self.k*self.gamma, len(episode)-idx-1)
                        X.append((episode[idx], episode[idx+step]))
                        y.append(0)
        return X, y

    def _train_r_model(self, X, y):
        for _ in range(5):
            indices = np.random.permutation(len(X))
            loss_total = 0
            for i in indices:
                obs1, obs2 = X[i]
                label = torch.tensor([y[i]], dtype=torch.float32).to(device)
                prob = self.r_model.get_label(obs1, obs2)
                loss = nn.functional.binary_cross_entropy(prob[:, 1], label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()