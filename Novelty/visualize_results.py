import pandas as pd
import matplotlib.pyplot as plt

def visualize_results():
    eval_freq = 10000
    total_timesteps = 100000

    df_simple_PPO = pd.read_csv('data/simple_ppo_rewards.csv', index_col=0)
    df_count_base = pd.read_csv('data/count_base_rewards.csv', index_col=0)
    df_episodic_curiousity = pd.read_csv('data/episodic_curiousity.csv', index_col=0)

    # Compute the mean and an exponentially smoothed version.
    for df in [df_simple_PPO, df_count_base, df_episodic_curiousity]:
        df["mean"] = df.iloc[:, 1:].mean(axis=1)
        df["mean_smoothed"] = df["mean"].ewm(alpha=0.1).mean()

    plt.figure()
    plt.plot(df_simple_PPO.index, df_simple_PPO['mean_smoothed'], label='Simple PPO')
    plt.plot(df_count_base.index, df_count_base['mean_smoothed'], label='Count-Based')
    plt.plot(df_episodic_curiousity.index, df_episodic_curiousity['mean_smoothed'], label='Episodic-Curiousity')

    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Average return over 5 runs (Evaluate over 10 episodes)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_results()
