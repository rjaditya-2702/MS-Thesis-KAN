import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

dtype = torch.float

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()
        
class Env_Runner:
    
    def __init__(self, env, agent, logger_folder):
        super().__init__()
        
        self.env = env
        self.agent = agent
        
        self.logger = Logger(f'{logger_folder}/training_info')
        self.logger.log("training_step, return")
        
        self.ob = self.env.reset()
        self.total_steps = 0

        self.ep_reward = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, steps):
        
        obs = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(steps):
            
            self.ob = torch.tensor(self.ob).to(self.device) # uint8
            action = self.agent.e_greedy(
                self.ob.to(dtype).unsqueeze(0) / 255) # float32+norm
            action = action.detach().cpu().numpy()
            
            obs.append(self.ob)
            actions.append(action)
            
            self.ob, r, done, info, additional_done = self.env.step(action)

            self.ep_reward += r
               
            if done: # real environment reset, other add_dones are for q learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps+step},{info["return"]}')
                
                print(f"Episode over. Reward: {self.ep_reward}")
                self.ep_reward = 0
            
            rewards.append(r)
            dones.append(done or additional_done)
            
        self.total_steps += steps
                                    
        return obs, actions, rewards, dones