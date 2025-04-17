import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from configs.hyperparams import HYPERPARAMS
from environments import CountBasedBonusWrapper, EpisodicCuriosityWrapper
from utils import EvalCallback, get_policy_kwargs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-16x16-v0")
    parser.add_argument("--method", type=str, default="base", 
                       choices=["base", "count_based", "episodic_curiosity"])
    parser.add_argument("--total_steps", type=int, default=100000)
    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = ResizeObservation(env, (84, 84))
    
    # Apply wrapper
    if args.method == "count_based":
        env = CountBasedBonusWrapper(env, HYPERPARAMS["count_based"])
    elif args.method == "episodic_curiosity":
        env = EpisodicCuriosityWrapper(env, HYPERPARAMS["episodic_curiosity"])
    
    # Initialize model
    model = PPO(
        env=env,
        verbose=1,
        **HYPERPARAMS["ppo"]
    )

    
    # Train with evaluation callback
    eval_callback = EvalCallback(env, eval_freq=10000, n_eval_episodes=10)
    model.learn(total_timesteps=args.total_steps, callback=eval_callback)
    
    # Save results
    np.save(f"results/{args.method}_rewards.npy", eval_callback.record_reward)
    model.save(f"models/{args.method}_ppo")

if __name__ == "__main__":
    main()