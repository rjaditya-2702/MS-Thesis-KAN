import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from Agent import DQN_Agent
from Atari_Wrapper import Atari_Wrapper
from Env_Runner import Env_Runner
from Experience_Replay import Experience_Replay

from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device('cuda') 
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    print("WHY IS ELSE GETTING EXECUTED?????")
dtype = torch.float

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

    global DEVICE 

    # create folder to save networks, csv, hyperparameter
    # folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    folder_name = 'Teacher'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
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

    eps_interval = start_eps-final_eps

    agent = DQN_Agent(in_channels, num_actions, start_eps).to(device=DEVICE)
    target_agent = DQN_Agent(in_channels, num_actions, start_eps).to(device=DEVICE)
    target_agent.load_state_dict(agent.state_dict())

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent, folder_name)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss()

    num_steps = 0
    num_model_updates = 0

    start_time = time.time()
    for num_steps in tqdm(range(0, total_steps, steps_rollout)):
        
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - ( eps_interval * num_steps/final_eps_frame))
        agent.set_epsilon(new_epsilon)
        
        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)
        
        # add
        # num_steps += steps_rollout
        
        # check if update
        if num_steps < min_replay_size_to_update:
            continue
        
        # update
        for update in range(4):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(DEVICE).to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(DEVICE)
            
            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(DEVICE).to(dtype)) / 255
            
            dones = torch.tensor([i[4] for i in minibatch]).to(DEVICE)
            
            #  *** double dqn ***
            # prediction
            
            Qs = agent(torch.cat([obs, next_obs]))
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size ,dim=0)
            
            obs_Q = obs_Q[range(minibatch_size), actions]
            
            # target
            
            next_obs_Q_max = torch.max(next_obs_Q,1)[1].detach()
            target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()
            
            target = rewards + gamma * target_Q * dones
            
            # loss
            loss = huber_loss(obs_Q, target)
            loss.backward()
            optimizer.step()
            
        num_model_updates += 1
         
        # update target network
        if num_model_updates%target_net_update == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # print time
        if num_steps%50000 < steps_rollout:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
        
        # save the dqn after some time
        if num_steps%save_model_steps < steps_rollout:
            torch.save(agent,f'{folder_name}/{env_name.replace("-","_")}.pt')

    env.close()
    

# 