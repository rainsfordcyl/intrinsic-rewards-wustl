import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.record_reward = []
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate()
            self.record_reward.append(mean_reward)
            print(f"Evaluation at step {self.n_calls}: Mean reward {mean_reward:.2f}")
        return True

    def _evaluate(self):
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = done or truncated
                
            rewards.append(episode_reward)
            
        return np.mean(rewards)