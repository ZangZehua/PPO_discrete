import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, device, obs_channels, fc_in_dim, action_dim, has_continuous_action_space, action_std_init):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 8, 4),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.GELU()
        )
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(fc_in_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
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

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        feature = self.conv(state)
        feature = feature.view(feature.shape[-1], -1)
        if self.has_continuous_action_space:
            action_mean = self.actor(feature)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(feature)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        feature = self.conv(state)
        feature = feature.view(feature.shape[-1], -1)
        if self.has_continuous_action_space:
            action_mean = self.actor(feature)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(feature)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)

        return action_logprobs, state_values, dist_entropy


if __name__ == "__main__":
    import torch.optim as optim
    device = torch.device("cuda:0")


    # def __init__(self, device, obs_channels, fc_in_dim, action_dim, has_continuous_action_space, action_std_init):
    net = ActorCritic(device, 1, 7*7*64, 6, False, None)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print(net)