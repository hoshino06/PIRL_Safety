# -*- coding: utf-8 -*-
"""
  Implementation of TD3-based PIRL with pytorch
"""
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
# Neural Networks (actor and critic)
########################################################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)		
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 32)
        self.l5 = nn.Linear(32, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.tanh(self.l1(sa))
        q1 = F.tanh(self.l2(q1))
        q1 = F.sigmoid(self.l3(q1))

        q2 = F.tanh(self.l4(sa))
        q2 = F.tanh(self.l5(q2))
        q2 = F.sigmoid(self.l6(q2))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.tanh(self.l1(sa))
        q1 = F.tanh(self.l2(q1))
        q1 = F.sigmoid(self.l3(q1))
        return q1

###########################################################################
# Reply buffer
###########################################################################

class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action     = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        # buffering
        self.state[self.ptr] = state
        self.action[self.ptr] = action
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
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
            )
    
    def __len__(self):
        return self.size

###########################################################################
# PIRL agent
###########################################################################

class PIRLagent(object):
    def __init__(self, state_dim, action_dim, max_action,
                 ## TD3 Options
                 ACTOR_LEARN_RATE   = 3e-4,
                 CRITIC_LEARN_RATE  = 3e-4,
         		 DISCOUNT           = 0.99,
                 REPLAY_MEMORY_SIZE = 5000, 
                 REPLAY_MEMORY_MIN  = 100,
                 MINIBATCH_SIZE     = 32,
                 EXPL_NOLISE        = 0.1,      # Std of Gaussian exploration noise
                 
                 TAU                = 0.005,
                 POLICY_NOISE       = 0.2, 
                 NOISE_CLIP         = 0.5,
                 POLICY_FREQ        = 2,
                 ## PINN options
                 CONVECTION_MODEL   = None,
                 DIFFUSION_MODEL    = None,
                 SAMPLING_FUN       = None, 
                 WEIGHT_PDE         = 1e-2, 
                 WEIGHT_BOUNDARY    = 1
                 ):
        
        # Neural Networks (Actor and Critic)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LEARN_RATE)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARN_RATE)

        # Replay Memory
        self.replay_memory = ReplayMemory(state_dim, action_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE

        # TD3 Options
        self.action_dim   = action_dim
        self.max_action   = max_action
        self.discount     = DISCOUNT
        self.EXPL_NOISE   = EXPL_NOLISE
        self.tau          = TAU
        self.policy_noise = POLICY_NOISE
        self.noise_clip   = NOISE_CLIP
        self.policy_freq  = POLICY_FREQ

        # Initialization
        self.policy_update_cnt = 0
		
    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
    def get_value(self, state):
        state  = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return self.critic.Q1(state, action).cpu().data.numpy().flatten()
        
    
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
    
        #####################################################################
        # Main loop
        #####################################################################
        for episode in iterator:
            
            ##########################
            # Reset
            ##########################
            state, is_done = env.reset(), False    
            episode_reward = 0
            episode_q0 = self.get_value(state)
         
            #######################################################
            # Iterate until episode ends 
            #######################################################
            while not is_done:
                
                # get action
                if len(self.replay_memory) < self.REPLAY_MEMORY_MIN:
                    action = self.max_action * ( np.random.rand(self.action_dim) -0.5 ) * 2
                else:
                    noise = np.random.normal(0, self.max_action * self.EXPL_NOISE, size=self.action_dim)
                    action = (self.get_action(state)+noise).clip(-self.max_action, self.max_action)
                
                # make a step
                next_state, reward, is_done = env.step(action)
                episode_reward += reward
        
                # train Q network
                self.replay_memory.add(state, action, next_state, reward, is_done)

                if len(self.replay_memory) > self.REPLAY_MEMORY_MIN:
                    self.update_step()
        
                # update current state
                state = next_state

            ###################################################
            # Log
            ###################################################
            if LOG_DIR: 
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Q0",     episode_q0,     episode)
                summary_writer.flush()
    
                if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                    
                    ckpt_path = summary_dir + f'/agent-{episode}'
                    torch.save({'actor-weights':  self.actor.state_dict(),
                                'critic-weights': self.critic.state_dict(),
                                }, ckpt_path)
    
    ####################################################################
    # Update of actor and critic
    ####################################################################
    def update_step(self):

        ###############################################
        # Calculate TD3 Critic loss
        ###############################################
        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.replay_memory.sample(self.MINIBATCH_SIZE)

        # Calculate target 
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
			
            next_action = (
                self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Compute critic loss
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ###############################################
        # Delayed policy updates
        ###############################################
        self.policy_update_cnt = (self.policy_update_cnt + 1) % self.policy_freq
        if self.policy_update_cnt == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
