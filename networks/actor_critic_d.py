import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_channels, fc_in_dim, action_dim):
        super().__init__()
        self.fc_in_dim = fc_in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 8, 4),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.GELU()
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        feature = self.conv(state)
        feature = feature.view(-1, self.fc_in_dim)
        action_probs = self.actor(feature)
        dist = Categorical(action_probs)

        action = dist.sample()  # 返回[1,1,action_dim]
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        feature = self.conv(state)
        feature = feature.view(-1, self.fc_in_dim)

        action_probs = self.actor(feature)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)

        return action_logprobs, state_values, dist_entropy

