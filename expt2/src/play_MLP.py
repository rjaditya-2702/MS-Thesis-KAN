# Reference: Marc Höftmann , DDQN-Atari-PyTorch
# GitHub. https://github.com/Hauf3n/DDQN-Atari-PyTorch
################################

import torch
import gym
from Atari_Wrapper import Atari_Wrapper
import time
import numpy as np

from Agent import DQN_Agent, KQN_Agent

from datetime import datetime
from config import ARGS as args

np.bool8 = np.bool_
dtype = torch.float

device = torch.device('cpu')

def play(env):
    env_name = env
    num_stacked_frames = args.stacked_frames
   
    raw_env = gym.make(env_name, render_mode = 'human')
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)

    # agent = DQN_Agent(num_stacked_frames, env.action_space.n, 0)
    agent = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location= torch.device('cpu'), weights_only=False)
    # agent.network.load_state_dict(st, strict=False, state_dict=st)
    agent.eval()
    agent.set_epsilon(0)

    steps = 50000
    ob = env.reset()
    imgs = []
    for _ in range(steps):
        ob1 = torch.tensor(ob, dtype=dtype)
        action = agent.e_greedy(ob1.unsqueeze(0).to(device) / 255)
        action = action.detach().cpu().numpy()
        # print()
        # print(action)
        # print()

        ob, _, done, info, _ = env.step(action)
        
        time.sleep(0.016)        
        if done:
            ob = env.reset()
            print(info)
        
        imgs.append(ob)
        
    env.close()
    return imgs

def main(env):
    images = play(env)

if __name__ == '__main__':
    main(env='PongNoFrameskip-v4')