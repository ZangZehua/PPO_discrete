import gym
import os
import numpy as np
import torch
import datetime
from tensorboardX.writer import SummaryWriter

from ppo import PPO
from common.config import Config
from common.atari_wrappers import wrap_deepmind


class Runner:
    def __init__(self):
        self.config = Config()
        print("=====" * 6)
        print("Run on {}, train environment: {}".format(self.config.device_name, self.config.env))
        print("=====" * 6)
        self.env = gym.make(self.config.env)
        self.env = wrap_deepmind(self.env)
        self.ppo_agent = PPO(self.config.device, self.config.obs_channels, self.config.fc_in_dim, self.config.action_dim,
                             self.config.lr, self.config.gamma, self.config.K_epochs, self.config.eps_clip)

    def get_state(self, observation):
        state = np.array(observation)
        state = state.transpose((2, 0, 1))
        state = torch.FloatTensor(state)
        return state.unsqueeze(0)

    def train(self):
        observation = self.env.reset()
        all_reward = []
        episode = 0
        episode_reward = 0
        start_time = datetime.datetime.now().replace(microsecond=0)
        step = 0
        reward_writer = SummaryWriter('runs/' + self.config.train_time + '/train_reward')
        while step < self.config.max_step:
            state = self.get_state(observation)
            action, action_logprob = self.ppo_agent.select_action(state)
            next_observation, reward, done, _ = self.env.step(action.item())
            # print(state.shape)

            # 存储state action action_logprob
            self.ppo_agent.buffer.states.append(state)
            self.ppo_agent.buffer.actions.append(action)
            self.ppo_agent.buffer.logprobs.append(action_logprob)
            self.ppo_agent.buffer.rewards.append(reward)
            self.ppo_agent.buffer.is_terminals.append(done)

            observation = next_observation
            episode_reward += reward

            if done:
                reward_writer.add_scalar("reward", episode_reward, episode)
                all_reward.append(episode_reward)
                episode_reward = 0
                episode += 1
                observation = self.env.reset()

            step += 1
            if step % self.config.upgrade_freq == 0:
                epoch_time = datetime.datetime.now().replace(microsecond=0)
                print("step: {}, reward: {}, episode:{}, train time: {}".format(
                    step, np.mean(all_reward[-10:]), episode, epoch_time - start_time))
                if not os.path.exists("saved/models/" + self.config.env):
                    os.mkdir("saved/models/" + self.config.env)
                self.ppo_agent.update()
                self.ppo_agent.save("saved/models/" + self.config.env + "/PPO-" + self.config.train_time + ".pth")

    def eval(self):
        print(self.env.reward())
        return


