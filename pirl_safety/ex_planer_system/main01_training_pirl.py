"""
  Training of PIRL agent for planer system
"""

import numpy as np
import random
import argparse
import torch

import sys, os
sys.path.append(os.pardir)
from agent import TD3, DQN

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
        next_state[0]  =  X1 + self.dt*( -X1**3- X2)
        next_state[1]  =  X2 + self.dt*( X1   + X2) + U
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



def convection_model(x_and_actIdx):

    x      = x_and_actIdx[:-1]
    actIdx = int(x_and_actIdx[-1]) 

    x1 = x[0]
    x2 = x[1]
    u  = [-1,0,1][actIdx]

    dxdt = np.array([-x1**3 -x2, 
                     x1 + x2 + u, 
                     -1 ])
 
    return dxdt    

def diffusion_model(x_and_actIdx):

    sig  = np.diag([0.2, 0.2, 0])
    diff = np.matmul( sig, sig.T )
 
    return diff

def sample_for_pinn():

    # Interior points    
    nPDE  = 32
    x_min, x_max = np.array([-1.5, -0.95, 0]), np.array([1.5, 0.95, 2.0])                
    X_PDE = x_min + (x_max - x_min)* np.random.rand(nPDE, 3)

    # Terminal boundary (at T=0 and safe)
    nBDini  = 32
    x_min, x_max = np.array([-1.5, -1.0, 0]), np.array([1.5, 1.0, 0])                
    X_BD_TERM = x_min + (x_max - x_min)* np.random.rand(nBDini, 3)

    # Lateral boundary (unsafe set)        
    nBDsafe = 32
    x_min, x_max = np.array([-1.5, 1.0, 0]), np.array([1.5, 1.0, 2.0])
    X_BD_LAT = x_min + (x_max - x_min)* np.random.rand(nBDsafe, 3)
    x2_sign  = np.sign(np.random.randn(nBDsafe) )
    X_BD_LAT[:,1] = X_BD_LAT[:,1] * x2_sign    
    
    return X_PDE, X_BD_TERM, X_BD_LAT



##########################################################
# Main     
##########################################################
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="TD3")         # Agent type (DQN or TD3)
    parser.add_argument("--seed", default=0, type=int)    # Sets PyTorch and Numpy seeds

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

    #########################################
    # TD3 agent
    #########################################
    if args.agent == "TD3":         
        agent = TD3.PIRLagent(state_dim, act_dim, max_act,
                              ### Learning rate
                              ACTOR_LEARN_RATE   = 1e-3,
                              CRITIC_LEARN_RATE  = 1e-3,
                              ### TD3 options
                              DISCOUNT           = 1,        
                              REPLAY_MEMORY_SIZE = int(1e4), 
                              REPLAY_MEMORY_MIN  = 1000,
                              MINIBATCH_SIZE     = 32,                       
                              TAU                = 0.005,
                              POLICY_NOISE       = 0.2, 
                              NOISE_CLIP         = 0.5,
                              POLICY_FREQ        = 4,
                              ### PINN options
                              CONVECTION_MODEL = convection_model,
                              DIFFUSION_MODEL  = diffusion_model,
                              SAMPLING_FUN     = sample_for_pinn, 
                              WEIGHT_PDE       = 1e-2, 
                              WEIGHT_BOUNDARY  = 1
                              )

    #########################################
    # DQN agent
    #########################################
    elif args.agent == "DQN":        
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
                              CONVECTION_MODEL = convection_model,
                              DIFFUSION_MODEL  = diffusion_model,
                              SAMPLING_FUN     = sample_for_pinn, 
                              WEIGHT_PDE       = 5e-3, 
                              WEIGHT_BOUNDARY  = 1, 
                              )

    print("--------------------------------------------")
    print(f"Agent: {args.agent}, Seed: {args.seed}")
    print("--------------------------------------------")

    
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

