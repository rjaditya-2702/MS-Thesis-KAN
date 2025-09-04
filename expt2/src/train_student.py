
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gym
import numpy as np
from Atari_Wrapper import Atari_Wrapper
from Agent import KQN_Agent
from config import ARGS as args
from tqdm import tqdm
from Env_Runner import Env_Runner
from Experience_Replay import Experience_Replay

np.bool8 = np.bool_
np.bool = np.bool_
dtype = torch.float

from tqdm import tqdm
from kan import KAN

import os

if torch.cuda.is_available():
    DEVICE = torch.device('cuda') 
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')


print(f"RUNNING ON {DEVICE}")

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class PureDistillationLoss(nn.Module):
    def __init__(self, temp=1.0):
        super(PureDistillationLoss, self).__init__()
        self.temp = temp

    def forward(self, student_logits, teacher_logits):
        # Apply softmax to the logits
        student_probs = F.softmax(student_logits / self.temp, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temp, dim=1)

        # Compute the distillation loss
        loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (self.temp ** 2)
        return loss

def make_transitions(obs, actions, rewards, dones):
    # observations are in uint8 format
    
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t+1],
                       int(not dones[t])))
        
    return tuples

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train(args):
    print("inside distill training")

    if not os.path.exists(f'./Student/'):
        os.mkdir('./Student/')

    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    replay_memory_size = args.replay_memory_size
    min_replay_size_to_update = args.replay_size_to_update
    lr = args.lr 
    gamma = args.gamma
    minibatch_size = args.minibatch_size
    steps_rollout = args.steps_rollout
    start_eps = args.start_eps
    final_eps = args.final_eps
    final_eps_frame = args.final_eps_frame
    total_steps = args.total_steps
    target_net_update = args.target_net_update
    save_model_steps = args.save_model_steps

    # init
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    student_model = KQN_Agent(in_channels, num_actions, 0).to(DEVICE)
    teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location= torch.device('cpu'), weights_only=False).to(DEVICE)
    teacher_model.eval()

    temp = 4
    distillation_criterion = PureDistillationLoss(temp=temp)

    runner = Env_Runner(env, student_model, 'Student')
    replay = Experience_Replay(replay_memory_size)

    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    total_steps = args.total_steps
    steps_rollout = args.steps_rollout

    ob = env.reset()
    print("starting training")
    ep_reward = 0

    for num_steps in tqdm(range(0, total_steps, steps_rollout)):
        optimizer.zero_grad()
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)

        if num_steps < min_replay_size_to_update:
            continue

        if total_steps % 150 == 0:
            torch.save(student_model.state_dict(),f'Student/{env_name.replace("-","_")}.pt')
            # print("model saved")
        
        for update in range(4):
            optimizer.zero_grad()
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(DEVICE).to(dtype)) / 255 
            
            Qs = student_model(obs)
            with torch.no_grad():
                target_Q = teacher_model(obs)
            
            # loss
            loss = distillation_criterion(Qs, target_Q)
            loss.backward()
            optimizer.step()

    torch.save(student_model.state_dict(),f'Student/{env_name.replace("-","_")}.pt')