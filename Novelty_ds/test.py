from stable_baselines3 import PPO
from environment import make_env

def test_model(model_path, num_episodes=50):
    env = make_env()
    model = PPO.load(model_path)
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print(f"Average reward over {num_episodes} episodes")