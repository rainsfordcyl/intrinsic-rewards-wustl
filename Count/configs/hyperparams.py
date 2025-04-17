# Hyperparameters for all components
import torch

HYPERPARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "ppo": {
    "policy": "CnnPolicy",
    "ent_coef": 0.005,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "policy_kwargs": {
        "activation_fn": torch.nn.ReLU,
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
    },
    
    "count_based": {
        "beta": 0.001,
        "state_dim": 147,  # 7x7x3
        "hash_dim": 56
    },
    
    "episodic_curiosity": {
        "alpha": 0.001,
        "beta": 1,
        "memory_size": 10,
        "k": 5,
        "gamma": 1.2,
        "train_interval": 30000,
        "embedding_dim": 512,
        "train_epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}