import torch
import os
import datetime


class Config:
    # Common config
    random_seed = 0  # set random seed if required (0 = no random seed)
    train_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S"))

    # CUDA config
    device = torch.device('cpu')
    device_name = "cpu"
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(device)

    # Env config
    env = "PongNoFrameskip-v4"
    has_continuous_action_space = False
    action_dim = 6

    # Network config
    obs_channels = 1
    fc_in_dim = 7 * 7 * 64

    lr = 0.001
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    # PPO train config
    max_step = 5000000
    update_timestep = 500 * 2  # update policy every n timesteps
    K_epochs = 40  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99
    action_std = None

    # print config
    upgrade_freq = 1000

    # log config
    log_dir = "runs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + '/' + env + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)



