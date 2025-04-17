import numpy as np
import torch

def set_seeds():
    np.random.seed(1)
    torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")