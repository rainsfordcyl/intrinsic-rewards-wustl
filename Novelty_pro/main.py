import random
import copy
import math
from collections import deque
import torch
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from gymnasium.core import Wrapper
from minigrid.wrappers import ImgObsWrapper

from utils import get_policy_kwargs
from Eval_Callback import Eval_Callback

np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

class SimHash:
    def __init__(self, state_emb, k):
        self.A = np.random.normal(0, 1, (k, state_emb))

    def hash(self, state):
        return str(np.sign(self.A @ np.array(state)).tolist())

class CountBasedBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.M = {}
        self.hash = SimHash(147, 56)
        self.beta = 0.001

    def _update_count_dict(self, h):
        pre_count = self.M[h] if h in self.M else 0
        self.M[h] = pre_count + 1

    def get_count(self, h):
        return self.M[h]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_flat = obs.flatten()
        h = self.hash.hash(obs_flat)
        self._update_count_dict(h)
        c = self.get_count(h)
        bonus = self.beta / math.sqrt(c)
        reward += bonus
        return obs, reward, terminated, truncated, info

class R_Model(nn.Module):
    def __init__(self):
        super(R_Model, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, 512)
        )
        self.classification = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def get_embedding(self, ob):
        ob = torch.tensor(ob.reshape(1,3,7,7), dtype=torch.float32).to(device)
        return self.embedding(ob)

    def get_label(self, ob_1, ob_2):
        emb1 = self.get_embedding(ob_1)
        emb2 = self.get_embedding(ob_2)
        cat = torch.cat((emb1, emb2), dim=1).to(device)
        return self.classification(cat)

    def get_reward(self, ob, memory):
        r = 0
        for m in memory:
            with torch.no_grad():
                prob = self.get_label(ob, m)[0][1].cpu()
            if prob > r:
                r = prob
        return r.item()

class EpisodicCuriosityBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.based_bonus = 0.001
        self.M = []
        self.eps = []
        self.max_length = 10
        self.step_retrained_model = 0
        self.r_model = R_Model().to(device)
        self.optimizer = optim.Adam(self.r_model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.model_trained = False
        self.beta = 1
        self.alpha = 0.001
        self.history = deque(maxlen=10)
        self.k = 5
        self.gamma = 1.2

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.eps.append(obs[0])
        self.M.append(obs[0])
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.eps.append(obs)
        if terminated or truncated:
            self.history.append(self.eps.copy())
            self.eps = []
            self.M = []
        self.step_retrained_model += 1
        if self.step_retrained_model == 30000:
            if len(self.history) != 0:
                X, y = self.create_training_data()
                self.train_r_model(X, y)
                self.model_trained = True
                self.step_retrained_model = 0
            else:
                self.step_retrained_model = 0
        if len(self.M) >= 2 and self.model_trained:
            b = self.r_model.get_reward(obs, self.M)
            b = self.alpha*(self.beta - b)
            reward += b
        if len(self.M) > self.max_length:
            if not any(np.array_equal(obs, arr) for arr in self.M):
                self.M.pop(random.randint(0, len(self.M) - 1))
                self.M.append(obs)
        else:
            if not any(np.array_equal(obs, arr) for arr in self.M):
                self.M.append(obs)
        return obs, reward, terminated, truncated, info

    def create_training_data(self):
        X, y = [], []
        for episode in self.history:
            for _ in range(30):
                idx, _ = random.choice(list(enumerate(episode)))
                max_step = min(self.k, len(episode) - 1 - idx)
                if max_step == 0:
                    continue
                step = 1 if max_step == 1 else random.randint(1, max_step)
                X.append([episode[idx], episode[idx+step]])
                y.append(1)
                if self.k*self.gamma <= len(episode) - 1 - idx:
                    step = random.randint(int(self.k*self.gamma), len(episode) - 1 - idx)
                    X.append([episode[idx], episode[idx+step]])
                    y.append(0)
        return X, y

    def train_r_model(self, X, y):
        for _ in range(5):
            indices = list(range(len(X)))
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            y_shuffled = [y[i] for i in indices]
            prob_stack = []
            for i in range(len(X)):
                prob = self.r_model.get_label(X_shuffled[i][0], X_shuffled[i][1])
                prob_stack.append(prob)
            prob_stack = torch.cat(prob_stack, dim=0)
            loss = self.criterion(prob_stack[:,1], torch.tensor(y_shuffled).float().to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
simple_ppo_reward = {}
for run in range(5):
    eval_callback = Eval_Callback(env, 10000, 10)
    policy = PPO("CnnPolicy", env, verbose=1, policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
    policy.learn(100000, callback=eval_callback)
    simple_ppo_reward[f"run_{run}"] = eval_callback.record_reward

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
count_base_reward = {}
for run in range(5):
    train_env = CountBasedBonusWrapper(env)
    eval_callback = Eval_Callback(env, 10000, 10)
    policy = PPO("CnnPolicy", train_env, verbose=1, policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
    policy.learn(100000, callback=eval_callback)
    count_base_reward[f"run_{run}"] = eval_callback.record_reward

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
episodic_curiosity = {}
for run in range(5):
    train_env = EpisodicCuriosityBonusWrapper(env)
    eval_callback = Eval_Callback(env, 10000, 10)
    policy = PPO("CnnPolicy", train_env, verbose=1, policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
    policy.learn(100000, callback=eval_callback)
    episodic_curiosity[f"run_{run}"] = eval_callback.record_reward

eval_freq = 10000
total_timesteps = 100000
row_idx = [i for i in range(0, total_timesteps, eval_freq)]
df_simple_PPO = pd.DataFrame.from_dict(simple_ppo_reward, orient='index').T
df_simple_PPO.index = row_idx
df_count_base = pd.DataFrame.from_dict(count_base_reward, orient='index').T
df_count_base.index = row_idx
df_episodic_curiosity = pd.DataFrame.from_dict(episodic_curiosity, orient='index').T
df_episodic_curiosity.index = row_idx

df_simple_PPO.to_csv('simple_ppo_rewards.csv')
df_count_base.to_csv('count_base_rewards.csv')
df_episodic_curiosity.to_csv('episodic_curiosity.csv')

df_simple_PPO = pd.read_csv('simple_ppo_rewards.csv')
df_count_base = pd.read_csv('count_base_rewards.csv')
df_episodic_curiosity = pd.read_csv('episodic_curiosity.csv')
dfs = [df_simple_PPO, df_count_base, df_episodic_curiosity]

for df in dfs:
    df["mean"] = df.iloc[:,1:].mean(axis=1)
    df["mean_smoothed"] = df["mean"].ewm(alpha=0.1).mean()

plt.plot(df_simple_PPO.iloc[:,0], df_simple_PPO['mean_smoothed'], label='Simple PPO')
plt.plot(df_count_base.iloc[:,0], df_count_base['mean_smoothed'], label='Count-Based')
plt.plot(df_episodic_curiosity.iloc[:,0], df_episodic_curiosity['mean_smoothed'], label='Episodic-Curiosity')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Average return over 5 runs (Evaluate over 10 episodes)')
plt.legend()
plt.show()
