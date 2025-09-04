import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import gym
import sys

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

def populate_replay_buffer(args):
    """
    Populate replay buffer with transitions from best games before training.
    
    Strategy:
    1. Play games using teacher model
    2. Keep track of the best cumulative reward
    3. Only add game transitions to buffer if cumulative reward exceeds best seen so far
    4. If buffer gets full, randomly drop transitions to make room for better games
    5. Save the buffer to disk
    """
    print("POPULATING REPLAY BUFFER WITH BEST GAMES")
    
    # Create Student directory if it doesn't exist
    if not os.path.exists('./Student/'):
        os.mkdir('./Student/')
    
    # Initialize environment and teacher model
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
    in_channels = num_stacked_frames
    num_actions = env.action_space.n
    
    # Load teacher model
    teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location=torch.device('cuda'), weights_only=False).cuda()
    teacher_model.eval()
    teacher_model.set_epsilon(0)
    
    # Initialize replay memory and environment runner
    replay_memory_size = args.replay_memory_size
    replay = Experience_Replay(replay_memory_size)
    
    best_reward = float('-inf')
    games_added = 0
    
    # Run until buffer is full
    pbar = tqdm(total=replay_memory_size)
    pbar.set_description("Filling buffer")
    
    while len(replay.memory) < replay_memory_size:
        # Reset environment variables for new game
        temp_buffer = []
        game_reward = 0
        game_done = False
        ob = env.reset()
        
        # Play a complete game
        while not game_done:
            # Get action from teacher
            ob_tensor = torch.tensor(ob)
            action = teacher_model.e_greedy(
                ob_tensor.cuda().to(dtype).unsqueeze(0) / 255)
            action = action.detach().cpu().numpy()
            
            # Take step in environment
            next_ob, reward, done, info, additional_done = env.step(action)
            game_reward += reward
            
            # Store transition
            temp_buffer.append((ob, action, reward))
            
            # Update state
            ob = next_ob
            game_done = done
        
        # Decide whether to add game to buffer
        if game_reward >= best_reward or len(replay.memory) < replay_memory_size * 0.1:  # Always add first 10% of games
            best_reward = max(best_reward, game_reward)
            
            # If buffer is full, make room by randomly dropping transitions
            if len(replay.memory) + len(temp_buffer) > replay_memory_size:
                # Calculate how many transitions we can add to memory
                space_needed = len(temp_buffer)
                free_space = replay_memory_size - len(replay.memory)

                if free_space < space_needed:
                    # Need to drop some existing transitions
                    to_drop = space_needed - free_space

                    # Randomly select indices to drop
                    drop_indices = np.random.choice(len(replay.memory), to_drop, replace=False)

                    # Create new memory without dropped transitions
                    new_memory = [replay.memory[i] for i in range(len(replay.memory)) if i not in drop_indices]
                    replay.memory = new_memory
                    replay.position = len(new_memory)

                # Instead of adding all transitions from temp_buffer, take only the most recent ones
                # that will fit in the available space
                space_available = replay_memory_size - len(replay.memory)
                transitions_to_add = temp_buffer[-space_available:]
            else:
                # We have enough space for all transitions
                transitions_to_add = temp_buffer

            # Add selected transitions to buffer
            for transition in transitions_to_add:
                if len(replay.memory) < replay_memory_size:
                    replay.memory.append(None)
                replay.memory[replay.position] = transition
                replay.position = (replay.position + 1) % replay_memory_size
            
            games_added += 1
            pbar.update(len(transitions_to_add))
            pbar.set_description(f"Filling buffer: {len(replay.memory)}/{replay_memory_size} transitions, Best reward: {best_reward:.1f}, Games added: {games_added}")
    
    pbar.close()
    print(f"Buffer populated with {len(replay.memory)} transitions from {games_added} games. Best reward: {best_reward:.1f}")
    
    # Save buffer to file using pickle instead of numpy
    import pickle
    dataset_path = f'./Student/dataset_{env_name.replace("-","_")}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(replay.memory, f)
    print(f"Dataset saved to {dataset_path}")
    del teacher_model
    del env
    del raw_env
    return replay

def load_dataset(args):
    """Load the dataset from a saved pickle file"""
    env_name = args.env
    dataset_path = f'./Student/dataset_{env_name.replace("-","_")}.pkl'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file {dataset_path} not found. Creating new dataset.")
        replay = populate_replay_buffer(args)
        return replay 
    
    print(f"Loading dataset from {dataset_path}")
    import pickle
    with open(dataset_path, 'rb') as f:
        memory = pickle.load(f)
    
    replay = Experience_Replay(args.replay_memory_size)
    replay.memory = memory
    replay.position = len(replay.memory) % replay.capacity
    
    print(f"Loaded {len(replay.memory)} transitions")
    return replay

def train(args):
    """Multi-GPU training using Data Parallelism for KANs"""
    print("TRAINING STUDENT WITH DATA PARALLELISM")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('./Student/'):
        os.mkdir('./Student')

    # Parse arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    lr = args.lr 
    temperature = 3
    minibatch_size = args.minibatch_size_kan
    total_epochs = 1  # Number of times to go through the entire replay buffer
    
    # GPU setup and monitoring
    num_gpus = torch.cuda.device_count()
    print(f"Training with {num_gpus} GPUs using DataParallel")

    # sys.exit(1)
    
    if num_gpus > 1:
        # Print GPU info
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("WARNING: Only one GPU detected. DataParallel requires multiple GPUs to be effective.")
        
    # Increase batch size if using multiple GPUs to utilize them effectively
    effective_minibatch_size = minibatch_size * num_gpus
    if num_gpus > 1:
        # Scale batch size with number of GPUs
        effective_minibatch_size = minibatch_size * num_gpus
        print(f"Scaling batch size to {effective_minibatch_size} for {num_gpus} GPUs")
    
    # Create function to monitor GPU usage during training
    def print_gpu_usage():
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    # Initialize environment
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)
    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    # Create model with DataParallel
    
    # student_model = nn.DataParallel(student_model)
    if os.path.exists(f'Student/{env_name.replace("-","_")}.pt'):
        print(f"Loading existing student model from {env_name.replace('-','_')}.pt")
        saved_model = torch.load(f'Student/{env_name.replace("-","_")}.pt')
        student_model = KQN_Agent(in_channels, num_actions, 0)
        student_model.load_state_dict(saved_model, strict=False)
        
    else:
        student_model = KQN_Agent(in_channels, num_actions, 0)

    # for module in student_model.modules():
    #     for param_name, param in list(module._parameters.items()):
    #         if param is not None and not isinstance(param, nn.Parameter):
    #             module._parameters[param_name] = nn.Parameter(param)

    student_model = student_model.cuda()
    student_model = nn.DataParallel(student_model, device_ids=list(range(num_gpus)))
    # student_model = student_model.cuda()
    
    # Load teacher model
    teacher_model = torch.load(f'Teacher/{env_name.replace("-","_")}.pt', map_location='cuda', weights_only=False)
    teacher_model.eval()
    teacher_model.set_epsilon(0)

    # Initialize optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # Load or create dataset
    replay = load_dataset(args)
    
    # Calculate number of batches per epoch
    buffer_size = len(replay.memory)
    batches_per_epoch = buffer_size // effective_minibatch_size
    total_batches = batches_per_epoch * total_epochs
    
    print(f"Training for {total_epochs} epochs with {batches_per_epoch} batches per epoch")
    print(f"Total batches: {total_batches}, Buffer size: {buffer_size}, Minibatch size: {effective_minibatch_size}")
    
    # Training loop - iterate over epochs
    global_step = 0
    loss_values = []
    for epoch in range(total_epochs):
        print(f"\nStarting epoch {epoch+1}/{total_epochs}")
        
        # Create dataset indices and shuffle them for this epoch
        indices = np.arange(buffer_size)
        np.random.shuffle(indices)
        
        # Process all batches in this epoch
        for batch_idx in tqdm(range(batches_per_epoch), desc=f"Epoch {epoch+1}"):
            # Zero gradients
            optimizer.zero_grad()
            
            # Get batch indices for this step
            start_idx = batch_idx * effective_minibatch_size
            end_idx = min(start_idx + effective_minibatch_size, buffer_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Get items from replay buffer using indices
            minibatch = [replay.memory[i] for i in batch_indices]
            
            # Convert numpy arrays to tensors before stacking
            obs_tensors = [torch.tensor(item[0], dtype=torch.float32) for item in minibatch]
            obs_tensor = torch.stack(obs_tensors) / 255
            
            # Log shapes to verify batch distribution across GPUs
            if batch_idx == 0 and epoch == 0:
                print(f"Batch shape: {obs_tensor.shape}")
                print_gpu_usage()
                
            # Forward pass through parallelized model
            print("start tensor", obs_tensor.device)
            print("model", next(student_model.parameters()).device)

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
            
            # Logging and saving (based on global step)
            if global_step % 1000 == 0:
                save_path = f'Student/{env_name.replace("-","_")}.pt'
                # Save original model state (not DataParallel wrapper)
                torch.save(student_model.state_dict(), save_path)
                loss_values.append(loss.item())
                print(f"\nStep {global_step}, Loss: {loss.item():.6f}")
                print_gpu_usage()  # Print GPU usage every 1000 steps
                
            global_step += 1
        
        # Print GPU usage at the end of each epoch
        print(f"\nEnd of epoch {epoch+1} GPU status:")
        print_gpu_usage()

    # Final model saving
    save_path = f'Student/{env_name.replace("-","_")}.pt'
    torch.save(student_model.state_dict(), save_path)
    print(f"Training completed after {global_step} steps, final model saved")

    # save loss plot
    import matplotlib.pyplot as plt
    plt.plot(loss_values)
    plt.xlabel('x 1000 steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'Student/loss_plot_{env_name.replace("-","_")}.png')
    plt.close()
    print(f"Loss plot saved as Student/loss_plot_{env_name.replace('-','_')}.png")