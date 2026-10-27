import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import transforms
import math
def layer_init(layer, std = math.sqrt(2), bias_const = 0.0):
    nn.init.orthogonal_(layer.weight, gain= std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden,act_dim), std= 0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, x):
        mu = self.net(x)
        
        std = torch.exp(self.log_std)
        base_dist = dist.Normal(mu, std)
        tanh_dist = dist.TransformedDistribution(
            base_dist,
            [transforms.TanhTransform(cache_size=1)]
        )

        return dist.Independent(base_dist, 1)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden = 64):
        super().__init__()
        
        self.obs_dim = obs_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden,1),std= 1.0)
        )
    
    def forward(self, x):
        return self.net(x)