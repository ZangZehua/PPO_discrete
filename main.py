import argparse
from runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False, help='train or evaluate, default eval')
    args = parser.parse_args()
    runner = Runner()
    if args.train:
        runner.train()
    else:
        runner.eval()


# print("==================================================")
# import gym, random, pickle, os.path, math, glob
# from gym import spaces
# import cv2
# cv2.ocl.setUseOpenCL(False)
# import pdb
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# from collections import deque
# from datetime import datetime
#
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
# from torch.distributions import MultivariateNormal
# from torch.distributions import Categorical
#
# from IPython.display import clear_output
# from tensorboardX import SummaryWriter
# print("import set")
# print("==================================================")