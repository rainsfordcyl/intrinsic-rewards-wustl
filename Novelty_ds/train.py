import pandas as pd
from stable_baselines3 import PPO
from environment import make_env
from wrappers import CountBasedBonusWrapper, EpisodicCuriosityBonusWrapper
from eval_callback import EvalCallback
from utils import get_policy_kwargs
from environment import create_vector_env

def train_agent(env_wrapper=None, num_runs=5, total_timesteps=100000):
    rewards = {}
    for run in range(num_runs):
        # Create properly wrapped vector environment
        env = create_vector_env(env_wrapper)
        
        # Create separate eval environment
        eval_env = create_vector_env()
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=10
        )

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=get_policy_kwargs(),
            ent_coef=0.005,
            verbose=1
        )
        
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        rewards[f"run_{run}"] = eval_callback.record_reward
    
    return pd.DataFrame(rewards)

if __name__ == "__main__":
    # Train Simple PPO
    df_simple = train_agent()
    df_simple.to_csv("simple_ppo_rewards.csv")

    # Train Count-Based
    df_count = train_agent(CountBasedBonusWrapper)
    df_count.to_csv("count_base_rewards.csv")

    # Train Episodic Curiosity
    df_episodic = train_agent(EpisodicCuriosityBonusWrapper)
    df_episodic.to_csv("episodic_curiosity.csv")