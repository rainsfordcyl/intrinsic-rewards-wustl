"""
utils.py
--------
This module contains utility functions used throughout the project.
It currently includes functions to:
    - Obtain policy keyword arguments for stable-baselines3 PPO.
    - Set a global seed for reproducibility.
    - Configure a basic logger.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import logging

def get_policy_kwargs() -> dict:
    """
    Returns a dictionary of keyword arguments for the policy network used by stable-baselines3 PPO.
    
    The returned dictionary configures the network architecture and the feature extractor:
      - activation_fn: Activation function for the network layers.
      - net_arch: A list defining the architecture for both policy (pi) and value function (vf) networks.
      - features_extractor_kwargs: Parameters for the features extractor (e.g., feature dimension).
    
    Returns:
        dict: Policy configuration parameters.
    """
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        features_extractor_kwargs=dict(features_dim=512)
    )
    return policy_kwargs

def set_global_seed(seed: int) -> None:
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Configures and returns a logger with the specified name.
    
    Args:
        name (str): The name of the logger.
    
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# For testing the module independently.
if __name__ == "__main__":
    logger = get_logger("utils")
    logger.info("Policy keyword arguments:")
    logger.info(get_policy_kwargs())
    logger.info("Setting global seed to 42.")
    set_global_seed(42)
