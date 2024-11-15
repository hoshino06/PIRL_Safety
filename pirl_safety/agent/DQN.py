# -*- coding: utf-8 -*-
""" 
  Implementation of DQN-based PIRL with pytorch
"""
import os
import copy
from   datetime import datetime
import numpy as np
from   tqdm import tqdm  # progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################
# Neural Networks (critic)
########################################################################

class Critic(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Critic, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(state_dim, 32), # input to layer1
            nn.Tanh(),
            nn.Linear(32, 32),        # layer1 to 2
            nn.Tanh(),
            nn.Linear(32, 32),        # layer2 to 3
            nn.Tanh(),
            nn.Linear(32, action_num), # layer 3 to out
            nn.Sigmoid()
            )
        
    def forward(self, state):       
        output = self.linear_stack(state)
        return output

###########################################################################
# Reply buffer
###########################################################################

class ReplayMemory(object):
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action_idx = np.zeros((max_size, 1), dtype=np.int32)
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))

    def add(self, state, action_idx, next_state, reward, done):
        # buffering
        self.state[self.ptr]      = state
        self.action_idx[self.ptr] = action_idx
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.LongTensor(self.action_idx[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
            )
    
    def __len__(self):
        return self.size

###############################################################################
# PIRL agent
###############################################################################

class PIRLagent:
    def __init__(self, state_dim, action_num, 
                 ## DQN options
                 CRITIC_LEARN_RATE   = 1e-3,
                 DISCOUNT            = 0.99, 
                 REPLAY_MEMORY_SIZE  = 5_000,
                 REPLAY_MEMORY_MIN   = 100,
                 MINIBATCH_SIZE      = 16, 
                 UPDATE_TARGET_EVERY = 5, 
                 EPSILON_INIT        = 1,
                 EPSILON_DECAY       = 0.998, 
                 EPSILON_MIN         = 0.01,
                 ## PINN options
                 PHYSICS_MODEL       = None, 
                 WEIGHT_PDE          = 1e-3, 
                 WEIGHT_BOUNDARY     = 1, 
                 HESSIAN_CALC        = True,
                 UNCERTAIN_PARAM     = None,
                 PARAM_LEARN_RATE    = None, 
                 ): 

        # Critic
        self.critic              = Critic(state_dim, action_num)
        self.critic_target       = copy.deepcopy(self.critic)
        self.critic_optimizer    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Replay Memory
        self.replay_memory     = ReplayMemory(state_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        
        # DQN Options
        self.actNum        =  action_num
        self.DISCOUNT      = DISCOUNT
        self.epsilon       = EPSILON_INIT 
        self.EPSILON_INI   = EPSILON_INIT
        self.EPSILON_MIN   = EPSILON_MIN
        self.EPSILON_DECAY = EPSILON_DECAY
        
        # PINN options
        self.physics_model    = PHYSICS_MODEL
        self.WEIGHT_PDE       = WEIGHT_PDE
        self.WEIGHT_BOUNDARY  = WEIGHT_BOUNDARY
        self.HESSIAN_CALC     = HESSIAN_CALC
        self.UNCERTAIN_PARAM  = UNCERTAIN_PARAM
        if not self.UNCERTAIN_PARAM == None:
            self.param_optimizer  = torch.optim.Adam([UNCERTAIN_PARAM], lr=PARAM_LEARN_RATE)
        
        # Initialization of variables
        self.target_update_counter = 0
                

    def get_qs(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)   
        return self.critic(state)
    
    
    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)   
        return self.critic(state).max()
    
    ####################################################################
    # Training loop
    ####################################################################
    def train(self, env, 
              EPISODES      = 50, 
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 1,
              RESTART_EP    = None
              ):
        
        ######################################
        # Prepare log writer 
        ######################################
        if LOG_DIR:
            summary_dir    = LOG_DIR+'/'+datetime.now().strftime('%m%d_%H%M')
            summary_writer = SummaryWriter(log_dir=summary_dir)
        if LOG_DIR and SHOW_PROGRESS:
            print(f'Progress recorded: {summary_dir}')
            print(f'---> $tensorboard --logdir {summary_dir}')        
    
        ##########################################
        # Define iterator for training loop
        ##########################################
        start = 0 if RESTART_EP == None else RESTART_EP
        if SHOW_PROGRESS:    
            iterator = tqdm(range(start, EPISODES), ascii=True, unit='episodes')        
        else:
            iterator = range(start, EPISODES)
    
        if RESTART_EP:
            self.epsilon = max(self.EPSILON_MIN, 
                               self.EPSILON_INI*np.power(self.EPSILON_DECAY,RESTART_EP))
    
        #######################################################################
        # Main loop
        #######################################################################
        for episode in iterator:
            
            ##########################
            # Reset
            ##########################
            state, is_done = env.reset(), False     
            episode_reward = 0
            episode_q0     = self.get_value(state)
        
            #######################################################
            # Iterate until episode ends 
            #######################################################
            while not is_done:
        
                # get action
                if np.random.random() > self.epsilon:
                    # Greedy action from Q network
                    action_idx = int( torch.argmax(self.get_qs(state)) )
                else:
                    # Random action
                    action_idx = np.random.randint(0, self.actNum)  
                action     = env.action_from_index(action_idx)
                
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
        
                # store experience and train Q network
                self.replay_memory.add(state, action_idx, next_state, reward, is_done)
                if len(self.replay_memory) > self.REPLAY_MEMORY_MIN:
                    self.update_critic()
        
                # update current state
                state = next_state
    
            ################################################
            # Update target Q-function and decay epsilon            
            ################################################
            self.target_update_counter += 1
            if self.target_update_counter > self.UPDATE_TARGET_EVERY:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.target_update_counter = 0
    
            if self.epsilon > self.EPSILON_MIN:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon  = max( self.EPSILON_MIN, self.epsilon)        
        
            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                if not self.UNCERTAIN_PARAM==None:
                    for i in range(len(self.UNCERTAIN_PARAM)):
                        summary_writer.add_scalar(f"Model Param{i}", self.UNCERTAIN_PARAM[i], episode)
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'critic-weights': self.critic.state_dict(),
                                'target-weights': self.critic_target.state_dict(),
                                }, 
                               ckpt_path)
   
    def update_critic(self):

        ###############################################
        # Calculate DQN loss
        ###############################################
        # Sample minibatch
        state, action_idx, next_state, reward, not_done = self.replay_memory.sample(self.MINIBATCH_SIZE)
        
        # Calculate target
        with torch.no_grad():
            next_value = self.critic_target(next_state).max(dim=-1,keepdim=True)[0]
            targetQ    = reward + not_done * self.DISCOUNT * next_value

        # DQN Loss (lossD)
        currentQ = self.critic(state).gather(1, action_idx)        
        loss     = F.mse_loss( currentQ, targetQ )        

        ########################################
        # Calculate PDE Loss
        ########################################
        if self.physics_model:
            # Samples for PDE
            X_PDE, X_ini, X_lat = self.physics_model.sampling()
            X_PDE = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            U_idx = self.critic(X_PDE).argmax(1, keepdim=True)
            if not self.UNCERTAIN_PARAM == None:
                f     = self.physics_model.convection(X_PDE,U_idx, self.UNCERTAIN_PARAM)
            else:
                f     = self.physics_model.convection(X_PDE,U_idx)
            dV_dx = torch.autograd.grad(self.critic(X_PDE).max(1).values.sum(), X_PDE, create_graph=True)[0]
            conv_term = ( dV_dx * f ).sum(1)

        # Diffusion term
        if self.physics_model and self.HESSIAN_CALC:
            # start_time_hess = datetime.now()
            diff_term = self.physics_model.diffusion(X_PDE, dV_dx)
            # end_time_hess = datetime.now()
            # elapsed_time = end_time_hess - start_time_hess
            # print("hess cal time:", elapsed_time)       
            loss += F.mse_loss(conv_term + diff_term, 
                               torch.zeros_like(conv_term))*self.WEIGHT_PDE             
        elif self.physics_model:
            loss += F.mse_loss(conv_term, 
                               torch.zeros_like(conv_term))*self.WEIGHT_PDE             
        
        ########################################
        # Calculate Boundary Loss (lossB)
        ########################################
        if self.physics_model:
            # termanal boundary (horizon = 0)
            V_ini  = self.critic(torch.tensor(X_ini, dtype=torch.float32)).max(1).values
            loss  += F.mse_loss( V_ini, torch.ones_like(V_ini))*self.WEIGHT_BOUNDARY
            
            # lateral boundary
            V_lat  = self.critic(torch.tensor(X_lat, dtype=torch.float32)).max(1).values
            loss  += F.mse_loss( V_lat, torch.zeros_like(V_lat))*self.WEIGHT_BOUNDARY
        
        #####################################
        # Update neural network weights 
        #####################################
        loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if not self.UNCERTAIN_PARAM==None:
            self.param_optimizer.step()
            self.param_optimizer.zero_grad()

        return # end: train_step


    def load_weights(self, ckpt_dir, ckpt_idx=None):

        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError("Directory '{}' does not exist.".format(ckpt_dir))

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums   = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt  = f'/agent-{check_nums.max()}'  
            ckpt_path    = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError("Check point 'agent-{}' does not exist.".format(ckpt_idx))

        checkpoint = torch.load(ckpt_path)
        self.critic.load_state_dict(checkpoint['critic-weights'])
        self.critic_target.load_state_dict(checkpoint['target-weights'])        
        #self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path    


