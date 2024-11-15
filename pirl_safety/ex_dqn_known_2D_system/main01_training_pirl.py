"""
  Training of PIRL agent for planer system
  (example with DQN and without uncertain parameter)
"""

import numpy as np
import random
import argparse
import torch

import sys, os
sys.path.append(os.pardir)
from agent import DQN

###########################
# RL Environment
###########################
class PlanerEnv(object):
    def __init__(self):
        self.dt = 0.1;
        self.ACTIONS = [-1, 0, 1]
        
    def reset(self):
        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.0  
        T   = 2.0
        self.state = np.array([X1, X2, T])
        return self.state

    def action_from_index(self, action_idx):
        return self.ACTIONS[action_idx] 

    def step(self, action):       
        # Current state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        U  = action
        # next state
        next_state = np.zeros_like(self.state)
        next_state[0]  =  X1 + self.dt*( - X1**3- X2)
        next_state[1]  =  X2 + self.dt*( X1   + X2 + U )
        next_state[2]  =  T -self.dt
        
        # Check terminal conditios 
        isTimeOver = (T <= self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe

        # Reward
        if done and (not isUnsafe):
            reward = 1
        else:
            reward = 0

        self.state = next_state
        
        return next_state, reward, done


###########################
# Physics Model:
# assuming no parameter uncertainty in this example
###########################
class PhysicsModel(object):    

    def __init__(self): 
        
        self.state_dim = 2 + 1 # state + horizon
        self.actions   = torch.tensor([-1, 0, 1])
        self.xi        = torch.tensor([0.0], requires_grad=True)

    def action_from_index(self, u_idx):
        return self.actions[u_idx]
        
    def convection(self, x, u):        
        output = torch.empty((x.size(0), self.state_dim))        
        output[:, 0] = x[:, 0]**3 - x[:, 1]
        output[:, 1] = x[:, 0] + x[:, 1] + u[:,0]
        output[:, 2] = -1            
        return output
    
    def diffusion(self, x, dV_dx):

        diff   = 0 
        sig    = [0.2, 0.2, 0]
        dV2dx2 = [torch.autograd.grad(dV_dx[:, i].sum(), x, retain_graph=True)[0][:,i]
                  for i in range(x.size(1))]
        for i in range(x.size(1)):
            diff += (sig[i]**2) * dV2dx2[i]        
        return diff 

    def sampling(self):        
        # Interior points    
        nPDE  = 100
        x_min, x_max = np.array([-1.5, -0.95, 0]), np.array([1.5, 0.95, 2.0])                
        X_PDE = x_min + (x_max - x_min)* np.random.rand(nPDE, 3)

        # Terminal boundary (at T=0 and safe)
        nBDini  = 32
        x_min, x_max = np.array([-1.5, -1.0, 0]), np.array([1.5, 1.0, 0])                
        X_TERM = x_min + (x_max - x_min)* np.random.rand(nBDini, 3)

        # Lateral boundary (unsafe set)        
        nBDsafe = 32
        x_min, x_max = np.array([-1.5, 1.0, 0]), np.array([1.5, 1.0, 2.0])
        X_LAT = x_min + (x_max - x_min)* np.random.rand(nBDsafe, 3)
        x2_sign  = np.sign(np.random.randn(nBDsafe) )
        X_LAT[:,1] = X_LAT[:,1] * x2_sign    

        return X_PDE, X_TERM, X_LAT


##########################################################
# Main     
##########################################################
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  default=0, type=int)    # Sets PyTorch and Numpy seeds

    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment
    rlenv      = PlanerEnv()
    state_dim  =  len(rlenv.reset())
    act_dim    = 1
    action_num = len(rlenv.ACTIONS)
    max_act    = 1
    
    # PhysicsModel
    model     = PhysicsModel()

    #########################################
    # DQN agent
    #########################################
    agent = DQN.PIRLagent(state_dim, action_num,
                          ## Learning rate
                          CRITIC_LEARN_RATE   = 5e-3,                              
                          ## DQN options
                          DISCOUNT            = 1, 
                          REPLAY_MEMORY_SIZE  = int(1e4), 
                          REPLAY_MEMORY_MIN   = 1000,
                          MINIBATCH_SIZE      = 32,                              
                          UPDATE_TARGET_EVERY = 5, 
                          ### Options for PINN
                          PHYSICS_MODEL       = model,
                          WEIGHT_PDE          = 1e-3, 
                          WEIGHT_BOUNDARY     = 1, 
                          HESSIAN_CALC        = True,
                          UNCERTAIN_PARAM     = None,
                          PARAM_LEARN_RATE    = 5e-4,
                          )
    
    #########################################
    # training
    #########################################
    
    LOG_DIR = 'logs/test'

    agent.train(rlenv, 
                EPISODES      = 3000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                )
