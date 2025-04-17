import numpy as np
import torch.nn as nn
from config import device

class SimHash:
    def __init__(self, state_emb, k):
        self.A = np.random.normal(0, 1, (k, state_emb))

    def hash(self, state):
        return str(np.sign(self.A @ np.array(state)).tolist())

class R_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3), nn.LeakyReLU(),
            nn.Flatten(), nn.Linear(64, 512))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Softmax(dim=1))

    def get_embedding(self, obs):
        obs = torch.FloatTensor(obs).permute(2,0,1).unsqueeze(0).to(device)
        return self.embedding(obs)

    def get_label(self, obs1, obs2):
        e1 = self.get_embedding(obs1)
        e2 = self.get_embedding(obs2)
        return self.classifier(torch.cat((e1, e2), dim=1))

    def get_reward(self, obs, memory):
        e = self.get_embedding(obs)
        similarities = [self.classifier(torch.cat((e, self.get_embedding(m)))) for m in memory]
        return max([s[0,1].item() for s in similarities])