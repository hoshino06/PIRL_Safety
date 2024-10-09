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
                 CONVECTION_MODEL    = None,
                 DIFFUSION_MODEL     = None,
                 SAMPLING_FUN        = None, 
                 WEIGHT_PDE          = 1e-3, 
                 WEIGHT_BOUNDARY     = 1, 
                 HESSIAN_CALC        = True,
                 ): 

        # Critic
        self.critic          = Critic(state_dim, action_num)
        self.critic_target       = copy.deepcopy(self.critic)
        self.critic_optimizer    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Replay Memory
        self.replay_memory = ReplayMemory(state_dim, REPLAY_MEMORY_SIZE)
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
        self.CONVECTION_MODEL = CONVECTION_MODEL
        self.DIFFUSION_MODEL  = DIFFUSION_MODEL
        self.SAMPLING_FUN     = SAMPLING_FUN
        self.WEIGHT_PDE       = WEIGHT_PDE
        self.WEIGHT_BOUNDARY  = WEIGHT_BOUNDARY
        self.HESSIAN_CALC     = HESSIAN_CALC
        
        # Initialization of variables
        self.target_update_counter = 0
                

    def get_qs(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)   
        return self.critic(state)
    
    
    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)   
        return self.critic(state).max()
    
    def get_epsilon_greedy_action(self, state):
        
        if np.random.random() > self.epsilon:
            # Greedy action from Q network
            action_idx = int( torch.argmax(self.get_qs(state)) )
        else:
            # Random action
            action_idx = np.random.randint(0, self.actNum)  
        return action_idx

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
    
        ###########################################################################
        # Main loop
        ###########################################################################
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
                action_idx = self.get_epsilon_greedy_action(state)
                action     = env.action_from_index(action_idx)
                
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
        
                # store experience and train Q network
                self.replay_memory.add(state, action_idx, next_state, reward, is_done)
                if len(self.replay_memory) > self.REPLAY_MEMORY_MIN:
                    self.update_step()
        
                # update current state
                state = next_state
    
            ################################################
            # Update target Q-function and decay epsilon            
            ################################################
            self.update_target()
            self.decay_epsilon()
        
            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'critic-weights': self.critic.state_dict(),
                                'target-weights': self.critic_target.state_dict(),
                                }, 
                               ckpt_path)
   
    def update_step(self):

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
        lossD  = F.mse_loss( currentQ, targetQ )        

        ########################################
        # Calculate PDE Loss
        ########################################
        # Samples for PDE
        X_PDE, X_BDini, X_BDlat = self.SAMPLING_FUN()
        #X_PDE = tf.Variable(X_PDE)        

        # Convection and diffusion coefficient
        with torch.no_grad():
            Qsa = self.critic(torch.tensor(X_PDE, dtype=torch.float))
            Uidx_PDE   = Qsa.argmax(1).numpy().reshape(-1, 1)               
        f =  np.apply_along_axis(self.CONVECTION_MODEL, 1, 
                                 np.concatenate([X_PDE, Uidx_PDE], axis=1) )
        A =  np.apply_along_axis(self.DIFFUSION_MODEL, 1, 
                                 np.concatenate([X_PDE, Uidx_PDE], axis=1))

        # PDE loss (lossP)
        if self.HESSIAN_CALC: 

            #from functorch import hessian
            
            X_PDE = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa   = self.critic(X_PDE)
            V     = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE, create_graph=True)[0]
            
            # jacobian_rows = [torch.autograd.grad(dV_dx, X_PDE, vec)[0]
            #          for vec in torch.eye(len(X_PDE[0]))]
            # return torch.stack(jacobian_rows)
                
        else: 
            
            X_PDE = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa   = self.critic(X_PDE)
            V     = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE, create_graph=True)[0]
                        
        #end_time_hess = datetime.datetime.now()
        #elapsed_time = end_time_hess - start_time_hess

        #print("calc_Hess:", elapsed_time)

        '''
        # check gradient implementation (for debug)
        print('\n V=', V)
        ##
        V_dx = tf.reduce_max( self.critic( X_PDE + [0.01, 0, 0]), axis=1)
        dV_dx_man = ( V_dx - V ) / 0.01
        print('dV_dx[:,0]=',  dV_dx[:,0])
        print('dV_dx[:,0] ~ ', dV_dx_man)
        ##
        V_dx2 = tf.reduce_max( self.critic( X_PDE - [0.01, 0, 0]), axis=1)
        HessV0_man = ( V_dx - 2.0*V + V_dx2 ) / ( (0.01)**2 )
        print(Hess[:,0])
        print(HessV0_man)
        '''                  

        ## Convection term
        conv_term = ( dV_dx * torch.tensor(f, dtype=torch.float32) ).sum(1)

        if self.HESSIAN_CALC:
            # Diffusion term            
            #diff_term = (1/2) * tf.linalg.trace( tf.matmul(A, HessV) )
            #diff_term = tf.cast(diff_term, dtype=tf.float32)
                          
            # lossP
            # lossP = tf.metrics.mean_squared_error(conv_term + diff_term, 
            #                                       np.zeros_like(conv_term) )
            lossP = torch.nn.functional.mse_loss( conv_term, 
                                                  torch.zeros_like(conv_term) )             

        else:
            # lossP
            lossP = torch.nn.functional.mse_loss( conv_term, 
                                                  torch.zeros_like(conv_term) )             
        
        ########################
        # Boundary loss (lossB)
        ########################
        # termanal boundary (\tau = 0)
        y_bd_ini = self.critic(torch.tensor(X_BDini, dtype=torch.float32)).max(1).values
        lossBini = torch.nn.functional.mse_loss( y_bd_ini, torch.ones_like(y_bd_ini) )
        
        # lateral boundary
        y_bd_lat = self.critic(torch.tensor(X_BDlat, dtype=torch.float32)).max(1).values
        lossBlat = torch.nn.functional.mse_loss( y_bd_lat, torch.zeros_like(y_bd_lat) )
        
        lossB = lossBini + lossBlat


        #####################################
        # Update neural network weights 
        #####################################
        Lambda = self.WEIGHT_PDE
        Mu     = self.WEIGHT_BOUNDARY
        loss = lossD + Lambda*lossP + Mu*lossB      

        loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        return # end: train_step


    def update_target(self):

        self.target_update_counter += 1
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.target_update_counter = 0

    def decay_epsilon(self):
        
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon  = max( self.EPSILON_MIN, self.epsilon)        


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
        self.critic.load_state_dict(checkpoint['weights'])
        self.target_model.load_state_dict(checkpoint['target-weights'])        
        self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path    


