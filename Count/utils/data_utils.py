import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def save_results(rewards, method_name, directory="results"):
    os.makedirs(directory, exist_ok=True)
    
    # Save as numpy array
    np.save(f"{directory}/{method_name}_rewards.npy", np.array(rewards))
    
    # Save as CSV with timesteps
    df = pd.DataFrame({
        'timestep': np.arange(len(rewards)) * 10000,
        'reward': rewards
    })
    df.to_csv(f"{directory}/{method_name}_rewards.csv", index=False)

def load_results(method_name, directory="results"):
    csv_path = f"{directory}/{method_name}_rewards.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def plot_results(methods, directory="results"):
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        df = load_results(method, directory)
        if df is not None:
            plt.plot(df['timestep'], df['reward'], label=method.replace('_', ' ').title())
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Exploration Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{directory}/comparison_plot.png")
    plt.close()