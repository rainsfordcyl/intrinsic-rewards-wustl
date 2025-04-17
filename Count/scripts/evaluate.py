import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

def evaluate_model(model_path, num_episodes=50, render=False):
    # Load environment
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # Load model
    model = PPO.load(model_path)
    
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            
            if render:
                img = env.render()
                plt.imshow(img)
                plt.pause(0.01)
                plt.clf()
        
        rewards.append(episode_reward)
        print(f"Episode {ep+1}/{num_episodes} - Reward: {episode_reward:.2f}")
    
    env.close()
    print(f"\nAverage reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Standard deviation: {np.std(rewards):.2f}")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RL Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render
    )