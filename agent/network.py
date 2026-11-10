"""
Actor and Critic networks for PPO.

Both are small MLPs with LeakyReLU activations and orthogonal initialization.
The actor outputs a diagonal Gaussian policy with a state-independent, learnable std.
"""
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import transforms
import math


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    """
    Orthogonal weight initialization with gain `std` and constant bias.
    Standard PPO choice: gain sqrt(2) for hidden layers, small gain for the policy head
    (near-zero initial actions) and 1.0 for the value head.
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Gaussian policy: obs -> Independent(Normal(mu(obs), std)).

    - mu comes from an MLP and is softly bounded to (-mu_max, mu_max)
    - std is a learnable parameter shared by all states (one log-std per action dimension)
    - actions are sampled in the "raw" (unsquashed) space; any squashing (e.g. tanh)
      is applied outside of the policy, by the caller
    """

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Two hidden layers; the last layer uses a tiny gain so that the initial mean
        # action is close to zero.
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01)
        )

        # State-independent log standard deviation (starts at log(1) = 0, i.e. std = 1).
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Soft bound for the mean of the policy.
        self.mu_max = 4
        # Hard bounds for log-std (std in [exp(-3), exp(0.5)]).
        self.log_std_bounds = (-3, 0.5)

    def forward(self, x):
        """Return the action distribution (a torch Distribution, not a tensor) for observation(s) x."""
        raw = self.net(x)

        # Smooth bound: ~identity near zero, saturates at +-mu_max.
        mu = self.mu_max * torch.tanh(raw / self.mu_max)
        
        log_std = self.log_std.clamp(*self.log_std_bounds)

        std = torch.exp(log_std)

        base_dist = dist.Normal(mu, std)

        # Treat the last dimension as one event: log_prob / entropy are summed over action dims.
        return dist.Independent(base_dist, 1)


class Critic(nn.Module):
    """State-value function V(obs): MLP with a scalar output."""

    def __init__(self, obs_dim, hidden=64):
        super().__init__()

        self.obs_dim = obs_dim

        # Same architecture as the actor; the value head uses gain 1.0.
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, 1), std=1.0)
        )

    def forward(self, x):
        """Return V(x) with shape (..., 1)."""
        return self.net(x)