import argparse
import train_MLP, train_KAN
import play_MLP, play_KAN
from config import ARGS

if __name__ == "__main__":
    
    # args = argparse.ArgumentParser()
    
    # set hyperparameter
    
    # args.add_argument('-lr', type=float, default=8e-5)
    # args.add_argument('-env', default='PongNoFrameskip-v4')
    # args.add_argument('-lives', type=bool, default=False)
    # args.add_argument('-stacked_frames', type=int, default=4)
    # args.add_argument('-replay_memory_size', type=int, default=250000)
    # args.add_argument('-replay_size_to_update', type=int, default=25000)
    # args.add_argument('-gamma', type=float, default=0.99)
    # args.add_argument('-minibatch_size', type=int, default=32)
    # args.add_argument('-steps_rollout', type=int, default=16)
    # args.add_argument('-start_eps', type=float, default=1)
    # args.add_argument('-final_eps', type=float, default=0.05)
    # args.add_argument('-final_eps_frame', type=int, default=500000)
    # args.add_argument('-total_steps', type=int, default=3000000)
    # args.add_argument('-target_net_update', type=int, default=625)
    # args.add_argument('-save_model_steps', type=int, default=1000000)
    # args.add_argument('-report', type=int, default=50000)
    # args.add_argument('-type', required=True, default = 'MLP')
    # args.add_argument('-mode', required=True)
    
    # arguments = args.parse_args()
    arguments = ARGS()
    print(f"PLAYING {arguments.env}")
    if arguments.mode == 'train_teacher':
        train_MLP.train(arguments)
    elif arguments.mode == 'play_teacher':
        play_MLP.play(arguments.env)    
    elif arguments.mode == 'train_student':
        train_KAN.train(arguments)
    elif arguments.mode == 'play_student':
        play_KAN.play(arguments.env)
    else:
        print("Bro what!")
