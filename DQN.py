import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random

class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        self.nn = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)   
        return out
        
class Q_learning(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, unit = hidden_unit, activation = F.relu):
        super(Q_learning, self).__init__()
        #print('type(hidden_layers)',type(hidden_layers))  # is a list
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList() # ModuleList() is different from sequential()
        self.in_channels = in_channels
        prev_layer = in_channels
        for hidden in hidden_layers:
            #print(unit(prev_layer, hidden, activation))
            #hidden_unit((nn): Linear(in_features=64, out_features=150, bias=True))
            #hidden_unit((nn): Linear(in_features=150, out_features=150, bias=True))
            self.hidden_units.append(unit(prev_layer, hidden, activation))
            #print(self.hidden_units)
            # ModuleList((0): hidden_unit((nn): Linear(in_features=64, out_features=150, bias=True)))
            # ModuleList(
            #   (0): hidden_unit(
            #     (nn): Linear(in_features=64, out_features=150, bias=True)
            #   )
            #   (1): hidden_unit(
            #     (nn): Linear(in_features=150, out_features=150, bias=True)
            #   )
            # )
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)
    
    def forward(self, x):
        print('x size xxxxxxx ', x.size()) # x is 1x64
        out = x.view(-1,self.in_channels).float() # out is 1x64
        #print('out size outoutoutout ', out.size())
        for unit in self.hidden_units:
            #print('unit size unitunitunit ', unit)
            # hidden_unit(
            #   (nn): Linear(in_features=64, out_features=150, bias=True)
            # )
            # hidden_unit(
            #   (nn): Linear(in_features=150, out_features=150, bias=True)
            # )

            out = unit(out)
        out = self.final_unit(out)
        return out


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'new_state', 'reward'))
Transition = namedtuple('Transition',
                        ('state', 'actionSpa','sa','action','new_state', 'reward'))


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #return random.sample(self.memory, batch_size)
        s_pos = random.randint(0, self.capacity - batch_size)
        return self.memory[s_pos:s_pos+batch_size]


    def __len__(self):
        return len(self.memory)  
    