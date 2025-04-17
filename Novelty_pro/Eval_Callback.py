import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class Eval_Callback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, verbose=0):
        super(Eval_Callback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.record_reward = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            ep_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                ep_reward = 0
                while not done and not truncated:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                ep_rewards.append(ep_reward)
            self.record_reward.append(np.mean(ep_rewards))
        return True
