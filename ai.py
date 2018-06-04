# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:57:45 2018

@author: Minkush
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# creating the architechture of NN

class Network(nn.Module):
    
    def __init__(self,input_size,nb_action):  # input_size = number of input neurons i.e no of neurons in i/p layer
    # nb_action : is output neurons(number)
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,30)  # this will create the full connection btw the neurons of 1st layer and the second hidden layer(30 neurons)
        self.fc2 = nn.Linear(30,nb_action)  # this will create the full connection btw the neurons of hidden layer and the o/p layer
        
    def forward (self, state): # state is the i/p param that this func takes it basically is the state of the map
        x = F.relu(self.fc1(state))  # this will give the activated hidden layer neurons
        q_values = self.fc2(x) # this will return the output neurons(q_values)
        return q_values
    
# implementing experience replay
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self,event):  # transition is basically a tuple of 4 elements old state, new state,last action ,last reward 
        self.memory.append(event)
        if len(self.memory) > self.capacity: # if the memory exceeds the capacity we will delete the oldest memory 
            del self.memory[0]
    
    def sample(self,batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x:Variable(torch.cat(x,0)),samples)  # this will create a list of batches where each batch will be pytorch variable
    
    
class Dqn():
    
    def __init__(self,input_size,nb_action,gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size,nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001 )
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self,state):
        probs = F.softmax(self.model(Variable(state,volatile= True))*100) # Temperature Parameter = 100
        # how certain the ai is to perform a action can be increased by increasing temperature param
        action = probs.multinomial() # this will give a random draw from the probability distribution
        return action.data[0,0]
    
    def learn(self,batch_state,batch_next_state,batch_reward,batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target  = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs,target) # loss function
        self.optimizer.zero_grad() # reinitialise the optimizer in each iteration
        td_loss.backward(retain_graph= True) # backpropagtes the errors
        self.optimizer.step() # updates the weight
        
    # an update function which will update the states,action once the car reaches a different state
    def update(self,reward,new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(self.last_reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    # will return mean of the rewards
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.0)
    
    # func to save the brain
    # we will be saving the weights and the optimmizer
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    },'last_brain.pth')  # the second arg of the save function will save this dictionary on last_brain_pth file
    
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> Loading the brain ...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done !")
        else:
            print("No Brain found!")
            