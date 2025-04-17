import torch
from torch import nn

class ReachabilityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64*3*3, 512)
        
        # Reachability classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1))
        
    def forward(self, obs1, obs2):
        emb1 = self.encoder(obs1)
        emb2 = self.encoder(obs2)
        combined = torch.cat([emb1, emb2], dim=1)
        return self.classifier(combined)