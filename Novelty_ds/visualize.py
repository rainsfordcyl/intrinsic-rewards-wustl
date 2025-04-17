import pandas as pd
import matplotlib.pyplot as plt

def load_and_process(filename):
    df = pd.read_csv(filename, index_col=0)
    df["mean"] = df.mean(axis=1)
    df["mean_smoothed"] = df["mean"].ewm(alpha=0.1).mean()
    return df

def plot_results():
    simple = load_and_process("simple_ppo_rewards.csv")
    count = load_and_process("count_base_rewards.csv")
    episodic = load_and_process("episodic_curiosity.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(simple.index, simple["mean_smoothed"], label="Simple PPO")
    plt.plot(count.index, count["mean_smoothed"], label="Count-Based")
    plt.plot(episodic.index, episodic["mean_smoothed"], label="Episodic Curiosity")
    plt.xlabel("Timestep"), plt.ylabel("Reward"), plt.legend()
    plt.title("Average Return Over 5 Runs")
    plt.show()

if __name__ == "__main__":
    plot_results()