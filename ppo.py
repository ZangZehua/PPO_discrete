import torch
import torch.nn as nn

from common.rollout_buffer import RolloutBuffer
# from networks.ActorCritic import ActorCritic
from networks.actor_critic_d import ActorCritic


class PPO:
    def __init__(self, device, obs_channels, fc_in_dim, action_dim,
                 lr, gamma, K_epochs, eps_clip):
        self.device = device

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_channels, fc_in_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(obs_channels, fc_in_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        return action, action_logprob

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        # old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        # old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_states = torch.stack(self.buffer.states, dim=0).squeeze(1).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).squeeze(1).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).squeeze(1).detach().to(self.device)
        # print(old_states.shape, old_actions.shape, old_logprobs.shape)


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))