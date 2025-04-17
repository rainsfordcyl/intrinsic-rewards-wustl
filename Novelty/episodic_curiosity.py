import random, copy
import math
import numpy as np
import torch
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from gymnasium.core import Wrapper
from minigrid.wrappers import ImgObsWrapper

from utils import get_policy_kwargs
from callbacks.Eval_Callback import EvalCallback

np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class R_Model(nn.Module):
    def __init__(self):
        super(R_Model, self).__init__()
        feature_output = 64
        # Embedding network (in the original paper a pretrained ResNet was used)
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_output, 512)
        )
        self.classification = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def get_embedding(self, ob):
        ob = torch.tensor(ob.reshape(1, 3, 7, 7), dtype=torch.float32).to(device)
        ob_emb = self.embedding(ob)
        return ob_emb

    def get_label(self, ob_1, ob_2):
        ob_1_emb = self.get_embedding(ob_1)
        ob_2_emb = self.get_embedding(ob_2)
        combined_embedding = torch.cat((ob_1_emb, ob_2_emb), dim=1).to(device)
        prob = self.classification(combined_embedding)
        return prob

    def get_reward(self, ob, M):
        max_reward = 0
        for ob_2 in M:
            with torch.no_grad():
                prob = self.get_label(ob, ob_2)
            prob = prob.to("cpu")
            value = prob[0][1]
            if value > max_reward:
                max_reward = value
        return max_reward.item()

class EpisodicCuriousityBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.based_bonus = 0.001
        self.M = []         # Memory storing observations for the current episode.
        self.eps = []
        self.max_length = 10
        self.step_retrained_model = 0                                    
        self.r_model = R_Model().to(device)
        self.optimizer = optim.Adam(self.r_model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.model_trained = False
        self.beta = 1
        self.alpha = 0.001
        self.history = deque(maxlen=10)  # Replay buffer for training.
        self.k = 5                                  
        self.gamma = 1.2  # Gap value between reachable and unreachable labels.

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

        # Train the reachability model periodically.
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
            bonus = self.r_model.get_reward(obs, self.M)
            bonus = self.alpha * (self.beta - bonus)
            reward += bonus
        
        # Update memory M.
        if len(self.M) > self.max_length:
            if not any(np.array_equal(obs, array) for array in self.M):
                self.M.pop(random.randint(0, len(self.M) - 1))
                self.M.append(obs)
        else:
            if not any(np.array_equal(obs, array) for array in self.M):
                self.M.append(obs)

        return obs, reward, terminated, truncated, info
    
    def create_training_data(self):
        X = []
        y = []
        for episode in self.history:
            for _ in range(30):
                episode_with_indices = list(enumerate(episode))
                index, _ = random.choice(episode_with_indices)
                max_step = min(self.k, len(episode) - 1 - index)
                if max_step == 1:
                    step = 1
                elif max_step == 0:
                    continue
                else:
                    step = random.randint(1, max_step)
                X.append([episode[index], episode[index + step]])
                y.append(1)
                if self.k * self.gamma > len(episode) - 1 - index:
                    continue
                else:
                    step = random.randint(int(self.k * self.gamma), len(episode) - 1 - index)
                    X.append([episode[index], episode[index + step]])
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
            loss = self.criterion(prob_stack[:, 1], torch.tensor(y_shuffled).float().to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train_episodic_curiousity():
    # Create environment.
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    episodic_curiousity = {}

    for run in range(5):
        train_env = EpisodicCuriousityBonusWrapper(env)
        test_env = env  
        eval_callback = EvalCallback(eval_env=env, eval_freq=10000, n_eval_episodes=10)
        policy = PPO(policy="CnnPolicy", env=train_env, verbose=1,
                     policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
        policy.learn(total_timesteps=100000, callback=eval_callback)
        episodic_curiousity[f"run_{run}"] = eval_callback.record_reward

    policy.save("pretrained_models/ppo_observation_episodic_curiousity")

    # Uncomment the block below to test the trained model.
    """
    policy = PPO.load("pretrained_models/ppo_observation_episodic_curiousity")
    obs, _ = test_env.reset()
    count = 1
    reward_list = []
    while count <= 50:
        action, _ = policy.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            reward_list.append(reward)
            obs, _ = test_env.reset()
            count += 1
    print(f"The average reward for 50 runs is: {np.mean(reward_list)}")
    test_env.close()
    """

    # Save rewards data to CSV.
    df_episodic_curiousity = pd.DataFrame.from_dict(episodic_curiousity, orient='index').T
    df_episodic_curiousity.index = [i for i in range(0, 100000, 10000)]
    df_episodic_curiousity.to_csv('data/episodic_curiousity.csv')
    print("Episodic curiousity training complete. Rewards saved to 'data/episodic_curiousity.csv'.")

if __name__ == "__main__":
    train_episodic_curiousity()
