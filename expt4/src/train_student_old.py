import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import gym

from Atari_Wrapper import Atari_Wrapper
from Agent import KQN_Agent
from config import ARGS as args
from Env_Runner import Env_Runner
from Experience_Replay import Experience_Replay

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

def train_single_gpu(args):
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

    student_model = KQN_Agent(in_channels, num_actions, 0).cuda()
    teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location= torch.device('cpu'), weights_only=False).cuda()
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("TRAINING STUDENT")
    for num_steps in tqdm(range(0, total_steps, steps_rollout)):
        teacher_model.set_epsilon(0)
        teacher_model.eval()

        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)

        # if num_steps < min_replay_size_to_update:
        #     continue
        running_loss = 0.0
        for update in range(1):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).cuda().to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch])


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


def train(args):
    """Multi-GPU training using Data Parallelism for KANs"""
    print("TRAINING STUDENT WITH DATA PARALLELISM")
    
    # Create output directory if it doesn't exist
    output_dir = '/home/rjaditya/KAN-1/Student'
    os.makedirs(output_dir, exist_ok=True)

    # Parse arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    lr = args.lr 
    temperature = 3
    replay_memory_size = args.replay_memory_size
    minibatch_size = args.minibatch_size_kan  # Already accounts for per-GPU batch size
    steps_rollout = args.steps_rollout
    total_steps = args.total_steps
    min_replay_size_to_update = args.replay_size_to_update
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Training with {num_gpus} GPUs using DataParallel")

    # Initialize environment
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    # Create model with DataParallel
    main_device = torch.device('cuda:0')
    student_model = KQN_Agent(in_channels, num_actions, 0)
    student_model = nn.DataParallel(student_model, device_ids=list(range(num_gpus)))
    student_model = student_model.cuda()
    
    # Load teacher model on main device
    teacher_path = f'/home/rjaditya/KAN-1/Teacher/{env_name.replace("-","_")}.pt'
    teacher_model = torch.load(teacher_path, map_location='cuda', weights_only=False)
    teacher_model.eval()
    teacher_model.set_epsilon(0)

    # Initialize optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # Initialize replay memory and environment runner
    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, teacher_model, 'Teacher')

    # Training loop
    for num_steps in tqdm(range(0, total_steps, steps_rollout)):
        # Collect data using teacher
        teacher_model.set_epsilon(0)
        teacher_model.eval()
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)

        # Skip updates until we have enough data
        # if num_steps < min_replay_size_to_update:
        #     continue
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Get batch (automatically scaled by DataParallel)
        minibatch = replay.get(minibatch_size)  # Total batch size
        
        # Prepare observations
        obs_tensor = (torch.stack([item[0] for item in minibatch])
                     .to(torch.float32).cuda()) / 255
        
        # Forward pass through parallelized model
        student_logits = student_model(obs_tensor)
        
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = teacher_model(obs_tensor)
        
        # Calculate loss
        student_log_probs = F.log_softmax(student_logits/temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits/temperature, dim=1)
        loss = F.kl_div(student_log_probs, teacher_prob, 
                       reduction='batchmean', log_target=False)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Logging and saving
        if num_steps % (steps_rollout * 10) == 0:
            save_path = f'{output_dir}/{env_name.replace("-","_")}.pt'
            # Save original model state (not DataParallel wrapper)
            torch.save(student_model.module.state_dict(), save_path)
            print(f"Step {num_steps}, Loss: {loss.item():.6f}")

    # Final model saving
    save_path = f'{output_dir}/{env_name.replace("-","_")}.pt'
    torch.save(student_model.module.state_dict(), save_path)
    print("Training completed, final model saved")