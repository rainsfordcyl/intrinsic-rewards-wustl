import random, copy
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

# Set seed and device.
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_simple_ppo():
    # Create environment.
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # Display environment info.
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    simple_ppo_reward = {}

    for run in range(5):
        eval_callback = EvalCallback(eval_env=env, eval_freq=10000, n_eval_episodes=10)
        policy = PPO(policy="CnnPolicy", env=env, verbose=1,
                     policy_kwargs=get_policy_kwargs(), ent_coef=0.005)
        policy.learn(total_timesteps=100000, callback=eval_callback)
        simple_ppo_reward[f"run_{run}"] = eval_callback.record_reward

    # Optionally, save the trained model.
    # policy.save("pretrained_models/simple_ppo")

    # Uncomment the block below to test the model.
    """
    policy = PPO.load("pretrained_models/simple_ppo")
    obs, _ = env.reset()
    count = 1
    reward_list = []
    while count <= 50:
        action, _ = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            reward_list.append(reward)
            obs, _ = env.reset()
            count += 1
    print(f"The average reward for 50 runs is: {np.mean(reward_list)}")
    env.close()
    """

    # Save rewards data to CSV.
    df_simple_PPO = pd.DataFrame.from_dict(simple_ppo_reward, orient='index').T
    df_simple_PPO.index = [i for i in range(0, 100000, 10000)]
    df_simple_PPO.to_csv('data/simple_ppo_rewards.csv')
    print("Simple PPO training complete. Rewards saved to 'data/simple_ppo_rewards.csv'.")

if __name__ == "__main__":
    train_simple_ppo()
