
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


def train(args):
    print("TRAINING STUDENT")
    if not os.path.exists(f'./Student/'):
        os.mkdir('./Student')

    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    lr = args.lr 
    minibatch_size = args.minibatch_size
    alpha = 0.35
    temperature = 3

    # init
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    student_model = KQN_Agent(in_channels, num_actions, 0).to(DEVICE)
    teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location= torch.device('cpu'), weights_only=False).to(DEVICE)
    teacher_model.eval()
    teacher_model.set_epsilon(0)

    replay_memory_size = args.replay_memory_size
    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, teacher_model, 'Teacher')
    minibatch_size = args.minibatch_size_kan
    steps_rollout = args.steps_rollout
    total_steps = args.total_steps
    min_replay_size_to_update = args.replay_size_to_update

    optimizer = optim.Adam(student_model.parameters(), lr = lr)

    print("TRAINING STUDENT")
    for num_steps in tqdm(range(0, total_steps, steps_rollout)):
        teacher_model.set_epsilon(0)
        teacher_model.eval()

        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)

        if num_steps < min_replay_size_to_update:
            continue
        running_loss = 0.0
        for update in range(1):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(DEVICE).to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(DEVICE)


            # print(rewards)

            student_logits = student_model(torch.cat([obs]))
            teacher_logits = teacher_model(torch.cat([obs]))

            student_log_probs = F.log_softmax(student_logits/temperature, dim=1, dtype=torch.float32)
            # teacher_log_probs = F.log_softmax(teacher_logits/temperature, dim=1, dtype=torch.float32)
            teacher_prob = F.softmax(teacher_logits/temperature, dim=1, dtype=torch.float32)
            # teacher_prob = torch.exp(teacher_log_probs)
            # kl_div = teacher_prob * (teacher_log_probs - student_log_probs)
            # divergent_loss = kl_div
            # divergent_loss = torch.sum(kl_div, dim=1).mean()
            # loss = divergent_loss + float(1/(np.exp(torch.sum(rewards.detach().cpu()))))
            loss = F.kl_div(student_log_probs, teacher_prob, reduction='batchmean', log_target=False)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print("Loss: {}".format(running_loss/4))
        # plotter.add_point(running_loss/4)
        # time.sleep(0.05)
        torch.save(student_model.state_dict(),f'Student/{env_name.replace("-","_")}.pt')
        print("Model saved")

