"""  Planer enviornment (2-dim example used in ACC2024 paper) 

    PlanerEnv class provides:
        1. reset method for initializing each episode
        2. step method that output (new_state, reward, is_done)
        3. convection model to specify convection term of HJB
        4. diffusion model to specify diffusion term of HJB
        5. sample_for_pinn method to generate samples for PINN

"""

__author__ = 'Hikaru Hoshino'
__email__ = 'hoshino@eng.u-hyogo.ac.jp'

import numpy as np

class PlanerEnv:

    dt = 0.1;
    ACTIONS = [-1, 0, 1]       
    
    def reset(self):

        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.5  #sign(r(2))*0.8 + sign(r(3))*0.35*rand;
        T   = 2.0
        self.state = np.array([X1, X2, T])
        
        return self.state

    def step(self, action_idx):       
        
        # Current state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        
        # New state
        U  = self.ACTIONS[action_idx]
        new_state = np.array([
                          X1 + self.dt*( -X1**3- X2),
                          X2 + self.dt*( X1   + X2  +U),
                           T - self.dt 
                     ])
        
        # Check terminal conditios 
        isTimeOver = (T <= self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe

        # Reward
        if done and (not isUnsafe):
            reward = 1
        else:
            reward = 0

        self.state = new_state
        
        return new_state, reward, done


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
    nPDE  = 8
    x_min, x_max = np.array([-1.5, -1.0, 0]), np.array([1.5, 1.0, 2.0])                
    X_PDE = x_min + (x_max - x_min)* np.random.rand(nPDE, 3)

    # Terminal boundary (at T=0 and safe)
    nBDini  = 8
    x_min, x_max = np.array([-1.5, -1.0, 0]), np.array([1.5, 1.0, 0])                
    X_BD_TERM = x_min + (x_max - x_min)* np.random.rand(nBDini, 3)

    # Lateral boundary (unsafe set)        
    nBDsafe = 8
    x_min, x_max = np.array([-1.5, 1.0, 0]), np.array([1.5, 1.0, 2.0])
    X_BD_LAT = x_min + (x_max - x_min)* np.random.rand(nBDsafe, 3)
    x2_sign  = np.sign(np.random.randn(nBDsafe) )
    X_BD_LAT[:,1] = X_BD_LAT[:,1] * x2_sign    
    
    return X_PDE, X_BD_TERM, X_BD_LAT
