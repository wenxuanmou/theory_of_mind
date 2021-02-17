import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
#from collections import namedtuple
#import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from convLSTM import ConvLSTM
import pdb
import torch.optim as optim

class charNet(nn.Module):

    def __init__(self,in_channels, out_channels, lstm_hidden):
        super(charNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3)
        self.convlstm = ConvLSTM(input_dim=8, hidden_dim=lstm_hidden, kernel_size=(3,3), num_layers=1, batch_first=False,bias=True, return_all_layers=False)
        self.fc = nn.Linear(lstm_hidden*3*3, out_channels) # h=3,w=3 after pooling


    def forward(self, x):
       
        t_,b_,c_,h,w = x.size() # size of x b_ for batch, t_ for time(num of sequences, c_ for channel),
        
        conv_x=[]
        for i in range(t_):
            
            temp=F.relu(self.conv1(x[i].view(b_,c_,h,w)))
            conv_x.append(temp)
        x=torch.stack(conv_x,0)
        
        _, x = self.convlstm(x)
        #pdb.set_trace()
        x = F.avg_pool2d(x[0][0],3)
        x = x.view(-1, self.num_flat_features(x))    
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



#Prediction net. In this experiment, we predict only next- step action (i.e. policy, ˆπ) 
#We spatialise echar,i, and concatenate with the query state. This is passed to a 2-layer convnet, 
#with 32 feature planes and ReLUs. This is followed by average pooling, then a fully-connected 
#layer to logits in R5, followed by a softmax.
class preNet(nn.Module):

    def __init__(self,in_channels, out_channels):
        super(preNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32,kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32,kernel_size=3)
        self.fc = nn.Linear(32*2*2, 5)
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(F.relu(self.conv2(x)),3)
        
        x = x.view(-1, self.num_flat_features(x))    
        x = self.fc(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#This main function was used for test the ToMnet only

if __name__ == '__main__':
    charnet = charNet(in_channels=64,out_channels=2,lstm_hidden=16)
    #optimizer = optim.SGD(charnet.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    x = torch.rand((32, 10, 64, 11, 11)) #t_=32,b_=10,c_=64,h=11,w=11 
    #target = Variable(torch.randn(32,2)).double()
    e_char = charnet(x)
    #pdb.set_trace()

    b_size = e_char.size(0)
    e_char1 = e_char[:,0]
    e_char2 = e_char[:,1]
    e_char1 = e_char1.view(b_size,1,1,1)
    e_char2 = e_char2.view(b_size,1,1,1)
    e_char1 = e_char1.expand(b_size,1,11,11)
    e_char2 = e_char2.expand(b_size,1,11,11)
    e_spatial = torch.cat((e_char1,e_char2),dim=1)



    # e_spatial = torch.zeros((b_size,2,11,11))
    # for i in range(e_char.size(0)):
    #     e_spatial[i,0,:,:]=e_char[i,0]
    #     e_spatial[i,1,:,:]=e_char[i,1]

 
    
    cur_s = torch.rand((10,3,11,11))
    y = torch.cat((e_spatial,cur_s),dim=1)
    prenet = preNet(in_channels=5, out_channels=5)
    #optimizer = optim.SGD([charnet.parameters(),prenet.parameters()], lr=0.01)
    optimizer = optim.SGD([{'params': charnet.parameters()},{'params': prenet.parameters(), 'lr': 0.01}], lr=1e-2)
    target = Variable(torch.randn(10,5)).double()
    output=prenet(y).double()

    #pdb.set_trace()


    #pdb.set_trace()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    #input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, 
    #bias=True, return_all_layers=False
    
    # charnet = charNet(64,2,16)
    # last_states = charnet(x)
    # h = last_states  # 0 for layer index, 0 for h index
    # loss_fn = torch.nn.MSELoss()
    
    # res = torch.autograd.gradcheck(loss_fn, (h.double(), target), eps=1e-6, raise_exception=True)
    # print('here',res)
    # print(type(h),h.size())


