
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

class DynamicPlot:
    def __init__(self, max_points=100, x_label="Time Step", y_label="Value", title="Dynamic Data Plot"):
        # Setup the figure and axis
        plt.ion()  # Turn on interactive mode here
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.grid(True)
        
        # Initialize data containers
        self.max_points = max_points
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        
        # Initialize the line
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        
        # Set initial axis limits
        self.ax.set_xlim(0, 10)
        # self.ax.set_ylim(-1.5, 1.5)
        
        # Counter for time steps
        self.current_step = 0
        
        # Show the plot immediately
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure the window appears
        
    def add_point(self, y_value):
        """Add a new data point to the plot"""
        # Increment the time step
        self.current_step += 1
        
        # Add data to deques
        self.x_data.append(self.current_step)
        self.y_data.append(y_value)
        
        # Dynamically adjust y-axis if needed
        if y_value > self.ax.get_ylim()[1] or y_value < self.ax.get_ylim()[0]:
            self.ax.set_ylim(min(self.y_data) - 1, max(self.y_data) + 1)
        
        # Adjust x-axis for moving window
        if len(self.x_data) >= 2:
            if len(self.x_data) >= self.max_points:
                # If we've reached max points, slide the window
                self.ax.set_xlim(self.current_step - self.max_points + 1, self.current_step + 5)
            else:
                # Otherwise just expand the window as needed
                self.ax.set_xlim(0, max(10, self.current_step + 5))
        
        # Update the line data
        self.line.set_data(list(self.x_data), list(self.y_data))
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Critical to ensure updates are visible
        
    def close(self):
        """Close the plot window"""
        plt.close(self.fig)
        plt.ioff()


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

    plotter = DynamicPlot(title="One Point at a Time")
    plt.ion()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# class DistillationLoss(nn.Module):
#     def __init__(self, alpha=0.5, temp=4.0):
#         super(DistillationLoss, self).__init__()
#         self.alpha = alpha  # Balance between distillation and regular CE loss
#         self.temp = temp    # Temperature for softening probability distributions
#         self.criterion = nn.CrossEntropyLoss()
        
#     def forward(self, outputs, labels, teacher_outputs):
#         # Regular cross entropy loss
#         ce_loss = self.criterion(outputs, labels)
        
#         # Distillation loss (KL divergence)
#         soft_targets = F.log_softmax(outputs / self.temp, dim=1)
#         soft_teacher = F.softmax(teacher_outputs / self.temp, dim=1)
#         distillation_loss = F.kl_div(soft_targets, soft_teacher, reduction='batchmean') * (self.temp ** 2)
        
#         # Total loss (weighted combination)
#         loss = self.alpha * ce_loss + (1.0 - self.alpha) * distillation_loss
#         return loss

# # Pure distillation loss (no label dependency)
# class PureDistillationLoss(nn.Module):
#     def __init__(self, temp=4.0):
#         super(PureDistillationLoss, self).__init__()
#         self.temp = temp
        
#     def forward(self, student_outputs, teacher_outputs):
#         # Ensure inputs are at least 2D for single sample case
#         if student_outputs.dim() == 1:
#             student_outputs = student_outputs.unsqueeze(0)
#         if teacher_outputs.dim() == 1:
#             teacher_outputs = teacher_outputs.unsqueeze(0)
            
#         # Apply temperature scaling
#         soft_student = F.log_softmax(student_outputs / self.temp, dim=1)
#         soft_teacher = F.softmax(teacher_outputs / self.temp, dim=1)
        
#         # KL divergence loss with safe reduction handling for any batch size
#         if student_outputs.size(0) == 1:
#             # For single samples, use 'sum' reduction (equivalent to 'batchmean' for batch size 1)
#             distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='sum') * (self.temp ** 2)
#         else:
#             # For regular batches, use 'batchmean'
#             distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temp ** 2)
        
#         return distillation_loss

# def train(args):
    # for epoch in tqdm(range(20)):
    #     ob = env.reset()
    #     student_model.train()
    #     running_loss = 0.0
    #     ep_steps = 0
    #     running_reward = 0.0
    #     for ep_step in tqdm(range(1000000)):
    #         x = torch.tensor(ob, dtype=dtype)
    #         optimizer.zero_grad()

    #         with torch.no_grad():
    #             teacher_logits = teacher_model(x.unsqueeze(0).to(DEVICE) / 255)
    #             y = torch.argmax(teacher_logits)
    #         student_logits = student_model(x.unsqueeze(0).to(DEVICE) / 255)

    #         # print(student_logits.shape)
            
    #         student_log_probs = F.log_softmax(student_logits/temperature, dim=1, dtype=torch.float32)
    #         teacher_log_probs = F.log_softmax(teacher_logits/temperature, dim=1, dtype=torch.float32)
    #         # print(student_log_probs)
    #         # print(teacher_log_probs)
            
    #         teacher_prob = torch.exp(teacher_log_probs)
    #         kl_div = teacher_prob * (teacher_log_probs - student_log_probs)
    #         divergent_loss = torch.sum(kl_div, dim=1).mean()

    #         ob, reward, done, info, _ = env.step(y)
    #         running_reward += reward

    #         # loss
    #         # loss = alpha*true_loss + (1-alpha)*divergent_loss
    #         loss = divergent_loss + 1/(np.exp(reward))
    #         # print(loss.item())

    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
            
    #         ep_steps += 1
    #         if done:
    #             break
    #     print("LOSS AFTER EPOCH{} = {}".format(epoch+1, running_loss/ep_steps))
    #     print("REWARD AFTER EPOCH (eposide) = {}".format(running_reward))
    #     torch.save(student_model.state_dict(),f'Student_simple_supervised/{env_name.replace("-","_")}.pt')
    #     print("Model saved")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# def train(args):
#     print("inside distill training")

#     if not os.path.exists(f'./Student/'):
#         os.mkdir('./Student/')

#     # arguments
#     env_name = args.env
#     num_stacked_frames = args.stacked_frames
#     replay_memory_size = args.replay_memory_size
#     min_replay_size_to_update = args.replay_size_to_update
#     lr = args.lr 
#     gamma = args.gamma
#     minibatch_size = args.minibatch_size
#     steps_rollout = args.steps_rollout
#     start_eps = args.start_eps
#     final_eps = args.final_eps
#     final_eps_frame = args.final_eps_frame
#     total_steps = args.total_steps
#     target_net_update = args.target_net_update
#     save_model_steps = args.save_model_steps

#     # init
#     raw_env = gym.make(env_name)
#     env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
#     in_channels = num_stacked_frames
#     num_actions = env.action_space.n

#     student_model = KQN_Agent(in_channels, num_actions, 0).to(DEVICE)
#     teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location= torch.device('cpu'), weights_only=False).to(DEVICE)
#     teacher_model.eval()

#     temp = 4
#     distillation_criterion = PureDistillationLoss(temp=temp)

#     runner = Env_Runner(env, student_model, 'Student')
#     replay = Experience_Replay(replay_memory_size)

#     optimizer = optim.Adam(student_model.parameters(), lr=lr)

#     total_steps = args.total_steps
#     steps_rollout = args.steps_rollout

#     ob = env.reset()
#     print("starting training")
#     ep_reward = 0

#     for num_steps in tqdm(range(0, total_steps, steps_rollout)):
#         optimizer.zero_grad()
#         obs, actions, rewards, dones = runner.run(steps_rollout)
#         transitions = make_transitions(obs, actions, rewards, dones)
#         replay.insert(transitions)

#         if num_steps < min_replay_size_to_update:
#             continue

#         if total_steps % 150 == 0:
#             torch.save(student_model.state_dict(),f'Student/{env_name.replace("-","_")}.pt')
#             # print("model saved")
        
#         for update in range(4):
#             optimizer.zero_grad()
#             minibatch = replay.get(minibatch_size)
            
#             # uint8 to float32 and normalize to 0-1
#             obs = (torch.stack([i[0] for i in minibatch]).to(DEVICE).to(dtype)) / 255 
            
#             Qs = student_model(obs)
#             with torch.no_grad():
#                 target_Q = teacher_model(obs)
            
#             # loss
#             loss = distillation_criterion(Qs, target_Q)
#             loss.backward()
#             optimizer.step()

#     torch.save(student_model.state_dict(),f'Student/{env_name.replace("-","_")}.pt')