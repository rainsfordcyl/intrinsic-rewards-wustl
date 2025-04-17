import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 5, verbose: int = 1):
        """
        :param eval_env: (gym.Env) The environment used for evaluation.
        :param eval_freq: (int) Number of steps between evaluations.
        :param n_eval_episodes: (int) Number of episodes to run for each evaluation.
        :param verbose: (int) Verbosity level.
        """
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.record_reward = []  # List to store mean reward at each evaluation

    def _init_callback(self) -> None:
        # Called once when training starts; reset record_reward list.
        self.record_reward = []

    def _on_step(self) -> bool:
        # Evaluate the agent every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                # Reset the evaluation environment.
                reset_result = self.eval_env.reset()
                # Gymnasium reset returns (obs, info); classic Gym returns just obs.
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result

                total_reward = 0.0
                done = False

                while not done:
                    # If observation is still wrapped in a tuple, extract it.
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    # Use the current policy (deterministic for evaluation) to get an action.
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.eval_env.step(action)
                    
                    # Check if the environment returns 5 values (Gymnasium) or 4 values (Gym)
                    if isinstance(step_result, tuple) and len(step_result) == 5:
                        obs, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, _ = step_result

                    total_reward += reward

                episode_rewards.append(total_reward)
            
            mean_reward = np.mean(episode_rewards)
            self.record_reward.append(mean_reward)
            
            if self.verbose > 0:
                print(f"\nEvaluation at step {self.n_calls}: mean reward = {mean_reward:.2f}")
        return True