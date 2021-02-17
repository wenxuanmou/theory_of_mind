from DQN import ReplayMemory, Transition
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch
import pdb

import torch.nn as nn
#from convLSTM2 import ConvLSTM
import torch.nn.functional as F
import os
import random

from scipy.stats import dirichlet
from ToMnet import charNet, preNet
import time



## Include the replay experience

epochs = 10
n_agents = 100
co_alpha = 1

#Creat the save folders
savedir = 'e_char/alpha1/actions/randomb_save'
if not os.path.exists(savedir):
    os.makedirs(savedir)
savedir_action = 'e_char/alpha1/actions/randomb_save' # save the action trajectories
if not os.path.exists(savedir_action):
    os.makedirs(savedir_action)
savedir_pre = 'e_char/alpha1/actions/randomb_save' # save the predicted results
if not os.path.exists(savedir_pre):
    os.makedirs(savedir_pre)

n_alpha = 5   
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 0.9 # epsilon for exploration or exploitation
input_size=11*11*5 # 11x11 is the size of the gridworld, 5 channels include walls, goals etc
mseloss = torch.nn.MSELoss()



for j in range(n_agents):
    time0 = time.time()
    pi = get_policy(co_alpha,n_alpha,rand_seed=j) # get the policy with dirichlet distribution 
    
    #new memory for each agent
    buffer = 32    
    memory = ReplayMemory(buffer) 

    # Network and the optimizer
    charnet = charNet(in_channels=10,out_channels=2,lstm_hidden=16)
    charnet = charnet.float()
    prenet = preNet(in_channels=7, out_channels=5)
    optimizer = optim.Adam([{'params': charnet.parameters()},{'params': prenet.parameters(), 'lr': 0.01}], lr=1e-2)
    actions_save = []
    BATCH_SIZE = random.randint(2,11) # N_past ~ U(2, 10)

    # loop over epochs
    for i in range(epochs):

        state = initGridPlayer(wall=False, n_obj=4)
        status = 1
        step = 0
        #while game still in progress
        new_state = state
        while(status == 1):
            v_state = Variable(torch.from_numpy(new_state))## variable of the state, BxCxHxW 
            action = select_action(pi)# action is one of five actions (stay, up, down, left, right)
            #print(dispGrid(new_state))
            #print('action is ', action)
            
            #Take action, observe new state S'
            new_state = makeMove(new_state, action)
            step +=1

            #spatialize the action and then concatenate with state
            action_spatio = action_spa(action)

            v_new_state = Variable(torch.from_numpy(new_state))# variable of new state
            v_action_spatio = Variable(torch.from_numpy(action_spatio)) # variable of spatilized action

            v_sa = torch.cat((v_state,v_action_spatio),2) # variable of state-action
            reward = getReward(new_state) # reward is not used in section 3.1, here is only used to end an episod
            memory.push(v_state.data, v_action_spatio.data, v_sa.data, action, v_new_state.data, reward)
            

            if (len(memory) < buffer): #if buffer not filled, add to it
                state = new_state
                if reward != -1: #if reached terminal state, update game status
                    break
                if step>31:
                    break
                else:
                    continue
            #print('**************      starting here     ****************')            
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions)) # tuple batchx(11x11x5=605)
            
            state_batch = Variable(torch.stack(batch.state)) #batchx11x11x5
            # action_batch =batch.action
            
            state_query = state_batch[-1,:,:,:].view(1,state_batch.shape[1],state_batch.shape[2],state_batch.shape[3])
            state_query = state_query.permute(0, 3, 1, 2)  #BHWC to BCHW

            #actionSpa_batch = Variable(torch.stack(batch.actionSpa))#batchx11x11x5
            action_batch = torch.LongTensor(batch.action).view(-1,1)#batchx1

            actions_save.append(action_batch.detach().numpy())
            #pdb.set_trace()

            action_query = action_batch[-1,:]

            #action_query = action_query.permute(0, 3, 1, 2)  #BHWC to BCHW

            sa_batch = Variable(torch.stack(batch.sa)[0:BATCH_SIZE-1,:,:,:])##batchx11x11x10

            reward_batch = Variable(torch.DoubleTensor(batch.reward)) #batch
            
            #non_final_mask = (reward_batch == -1) #40
            

            sa_batch = sa_batch.permute(0, 3, 1, 2)  #BHWC to BCHW
            sa_batch = sa_batch.view(sa_batch.shape[0],1,sa_batch.shape[1], sa_batch.shape[2],sa_batch.shape[3]) # BCHW to T(=B)B(=1)CHW
            
            e_char = charnet(sa_batch.float())
            
            np.save(savedir+'/agent'+str(j),e_char.detach().numpy())

            b_size = e_char.size(0)
            e_char1 = e_char[:,0]
            e_char2 = e_char[:,1]
            e_char1 = e_char1.view(b_size,1,1,1)
            e_char2 = e_char2.view(b_size,1,1,1)
            e_char1 = e_char1.expand(b_size,1,11,11)
            e_char2 = e_char2.expand(b_size,1,11,11)
            e_spatial = torch.cat((e_char1,e_char2),dim=1)

            criterion = nn.CrossEntropyLoss()
            #cur_s = torch.rand((32,3,11,11))
            input = torch.cat((e_spatial,state_query.float()),dim=1)

            

            y = prenet(input)
            tmp = F.softmax(y)
            tmp = tmp.detach().numpy()[0]
            msel = np.sum((tmp-pi)*(tmp-pi))/(pi.shape[0])#mseloss(tmp,pi)
            #print(tmp, ' --- action is ---- ', action_query, 'mse -- ', msel)
            
            loss = criterion(y,action_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if reward != -1:
                #pdb.set_trace()
                status = 0
            if step >31:
                #print('here the step is bigger than 31')
                break
        if epsilon > 0.1:
            epsilon -= (1/epochs)

        if (i+1) == epochs:
            gt_pred=[pi,tmp,action_query.detach().numpy()[0], np.argmax(tmp), (action_query.detach().numpy()[0]) == (np.argmax(tmp))] # gt_policy, pred_policy, gt_action, pred_action
            np.save(savedir_pre+'/gt_pre_agent'+str(j),gt_pred)
            print('saving for  agent ', j )
            
        print('agent ', j, ' epoch ', i+1, ' --action gt is -- ', action_query, '-- action pred is -- ', np.argmax(tmp), 'mse -- ', msel)

    np.save(savedir_action+'/action_agent'+str(j),actions_save)
    time1 = time.time()
    print('running time for agent ', j, ' is ', time1-time0 )