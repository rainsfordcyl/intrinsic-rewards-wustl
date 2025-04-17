import random, copy
import math
import numpy as np
import torch
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from gymnasium.core import Wrapper
from minigrid.wrappers import ImgObsWrapper

from utils import get_policy_kwargs
from callbacks.Eval_Callback import EvalCallback

np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimHash(object):
    def __init__(self, state_emb, k):
        self.A = np.random.normal(0, 1, (k, state_emb))

    def hash(self, state):
        # Compute the sign of the dot product and convert to a string key.
        hash_key = str(np.sign(self.A @ np.array(state)).tolist())
        return hash_key

class CountBasedBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.M = {}  # Memory counting dictionary.
        self.hash = SimHash(147, 56)
        self.beta = 0.001

    def _update_count_dict_(self, hash_key):
        pre_count = self.M.get(hash_key, 0)
        new_count = pre_count + 1
        self.M[hash_key] = new_count

    def get_count(self, hash_key):
        return self.M[hash_key]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_flatten = obs.flatten()
        hash_key = self.hash.hash(obs_flatten)
        self._update_count_dict_(hash_key)
        new_count = self.get_count(hash_key)
        bonus = self.beta / math.sqrt(new_count)
        reward += bonus
        return obs, reward, terminated, truncated, info

def train_count_based_bonus():
    # Create environment.
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    count_base_reward = {}

    for run in range(5):
        train_env = CountBasedBonusWrapper(env)
        test_env = env  # Original environment for evaluation.
        eval_callback = EvalCallback(eval_env=env, eval_freq=10000, n_eval_episodes=10)
        policy = PPO(policy="CnnPolicy", env=train_env, verbose=1,
                     policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
        policy.learn(total_timesteps=100000, callback=eval_callback)
        count_base_reward[f"run_{run}"] = eval_callback.record_reward

    # Optionally, save the trained model.
    policy.save("pretrained_models/ppo_observation_count_bonus")

    # Uncomment the block below to test the model.
    """
    policy = PPO.load("pretrained_models/ppo_observation_count_bonus")
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
    df_count_base = pd.DataFrame.from_dict(count_base_reward, orient='index').T
    df_count_base.index = [i for i in range(0, 100000, 10000)]
    df_count_base.to_csv('data/count_base_rewards.csv')
    print("Count-based bonus training complete. Rewards saved to 'data/count_base_rewards.csv'.")

if __name__ == "__main__":
    train_count_based_bonus()
