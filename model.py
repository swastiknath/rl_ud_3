'''
Author : Swastik Nath.
UDACITY DEEP REINFORCEMENT LEARNING NANODEGREE.
CAPSTONE 3: COLLABORATION AND COMPETITION.
ACTOR-CRITIC NEURAL NETWORK.
IMPLEMENTATION IN COURTESY OF UDACITY.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    '''
    Providing Limits for the values of input layers for uniform parameter weights distribution.
    Params:
    layer: Neural Network layer, whose parameter weights limit to find.
    '''
    in_size = layer.weight.data.size()[0]
    lim = 1./np.sqrt(in_size)
    return (-lim, lim)

class Actor(nn.Module):
    '''
    Actor Deep Neural Network Architecture 
    Works by estimating the Policy Gradients.
    '''
    
    def __init__(self, state_size, action_size, seed=0, fc1_size=128, fc2_size=128):
        '''
        Params:
        state_size: Size of the States of the Environment
        action_size: Size of the Action Space of the Environment.
        seed :  Random Seed for reproducibility.
        fc1_size: Size of the Fully Connected Layer 1.
        fc2_size: Size of the Fully Connected Layer 2.
        '''
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size=state_size
        self.action_size=action_size
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.reset_params()
        
    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        '''
        Feed-Forward Behaviour of the Neural Network:
        Params:
        state: State of the Environment parsed as Float Tensor
        '''
        if len(state) == self.state_size:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
class Critic(nn.Module):
    '''
    Critic Neural Network Acrhitecture.
    Works by Evaluating the Action-Value. 
    '''
    def __init__(self, state_size, action_size, seed=0, fc1_size=128, fc2_size=128):
        '''
        Params:
        state_size: Size of the States of the Environment
        action_size: Size of the Action Space of the Environment.
        seed :  Random Seed for reproducibility.
        fc1_size: Size of the Fully Connected Layer 1.
        fc2_size: Size of the Fully Connected Layer 2.
        '''
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size=state_size
        self.action_size=action_size
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size+action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.reset_params()
        
    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        '''
        Feed-Forward Behaviour of the Neural Network:
        Params:
        state: State of the Environment parsed as Float Tensor
        action: Action Taken on the Correspoing State as prescribed by Actor's Policy.
        '''
        x1 = F.relu(self.fc1(state))
        x1 = self.bn1(x1)
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)