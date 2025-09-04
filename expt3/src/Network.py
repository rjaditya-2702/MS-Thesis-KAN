import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from kan import KAN

if torch.cuda.is_available():
    device = torch.device('cuda') 
elif torch.backends.mps.is_available():
    device = torch.device('mps')
dtype = torch.float

class DQN(nn.Module):
    # nature paper architecture
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        actions = self.network(x)
        return actions

class KQN(nn.Module):
    # Architecture matching the saved Student model
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        
        network = [
            torch.nn.Conv2d(in_channels, 10, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(10, 15, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(15, 20, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Sequential(
                # nn.Linear(20*4*4, 20),
                KAN(width=[20*4*4, 8, 8, num_actions], grid = 15, k = 3)
            )
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        actions = self.network(x)
        return actions