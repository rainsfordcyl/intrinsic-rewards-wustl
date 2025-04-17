import random
import numpy as np
import torch
from collections import deque
from torch import nn, optim
from gymnasium.core import Wrapper


class EpisodicCuriosityWrapper(Wrapper):
    def __init__(self, env, params):
        super().__init__(env)
        self.params = params
        self.device = params["device"]
        self.episode_buffer = []
        self.memory = deque(maxlen=params["memory_size"])
        self.history = deque(maxlen=10)
        
        # Initialize network and optimizer
        self.r_net = ReachabilityNetwork().to(self.device)
        self.optimizer = optim.Adam(self.r_net.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.train_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store transition
        self.episode_buffer.append(obs.copy())
        
        # Add bonus if model is trained
        if len(self.memory) > 1 and self.train_step > 0:
            with torch.no_grad():
                bonus = self._calculate_bonus(obs)
            reward += self.params["alpha"] * (self.params["beta"] - bonus)
        
        # Handle episode end
        if terminated or truncated:
            self.history.append(self.episode_buffer.copy())
            self.episode_buffer = []
            
        # Train network periodically
        self.train_step += 1
        if self.train_step % self.params["train_interval"] == 0:
            if len(self.history) > 0:
                self._train_network()
                
        # Update memory
        self._update_memory(obs)
        
        return obs, reward, terminated, truncated, info

    def _calculate_bonus(self, obs):
        obs_tensor = torch.FloatTensor(obs).permute(2,0,1).unsqueeze(0).to(self.device)
        max_prob = 0.0
        
        for mem_obs in self.memory:
            mem_tensor = torch.FloatTensor(mem_obs).permute(2,0,1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = self.r_net(mem_tensor, obs_tensor)[0,1].item()
            max_prob = max(max_prob, prob)
            
        return max_prob

    def _update_memory(self, obs):
        if not any(np.array_equal(obs, m) for m in self.memory):
            if len(self.memory) >= self.params["memory_size"]:
                self.memory.pop(random.randint(0, len(self.memory)-1))
            self.memory.append(obs.copy())

    def _train_network(self):
        # Create training data
        X, y = [], []
        for episode in self.history:
            for _ in range(30):
                i = random.randint(0, len(episode)-1)
                j = random.randint(i+1, min(i+self.params["k"], len(episode)-1))
                X.append((episode[i], episode[j]))
                y.append(1)
                
                # Negative sample
                j = random.randint(i + int(self.params["k"] * self.params["gamma"]), len(episode)-1)
                X.append((episode[i], episode[j]))
                y.append(0)
                
        # Train network
        for _ in range(self.params["train_epochs"]):
            indices = np.random.permutation(len(X))
            for idx in indices:
                obs1, obs2 = X[idx]
                label = y[idx]
                
                obs1_t = torch.FloatTensor(obs1).permute(2,0,1).unsqueeze(0).to(self.device)
                obs2_t = torch.FloatTensor(obs2).permute(2,0,1).unsqueeze(0).to(self.device)
                
                pred = self.r_net(obs1_t, obs2_t)
                loss = self.criterion(pred[:,1], torch.FloatTensor([label]).to(self.device))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()