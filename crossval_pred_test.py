#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from cust_dataset import CustDataset
from network import Network
from torch.utils import *

from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.model_selection import ShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
import pandas as pd


# In[2]:


# torch.set_default_dtype(torch.float32)
# parser = argparse.ArgumentParser()
# parser.add_argument('job_id',type=str)
# args = parser.parse_args()
# print(args.job_id)
# print('number of gpus ',torch.cuda.device_count())


# In[3]:


# directory = args.job_id
# parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
# path = os.path.join(parent_directory,directory)
# model_save_path = os.path.join(path,'models_fold')

# if not os.path.exists(path):
#     os.mkdir(path)
#     os.mkdir(model_save_path)


# In[4]:


torch.manual_seed(52)
num_workers = 4
batch_size = 5
valid_size = 0.20
test_size = 0.10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# In[5]:


train_data = CustDataset(transform = 
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]),train=True)

valid_data = CustDataset(train=False,valid=True)

test_data = CustDataset(train=False,valid=False)


# In[6]:


# get filtered variables
vars = valid_data.vars


# In[7]:


# Prepare for k-fold
sss = ShuffleSplit(n_splits=5,test_size=0.2,random_state=52)
learning_rate = 0.00001
fold = 1


# In[8]:


next(sss.split(train_data.train_idx))


# In[9]:


for train_idx, valid_idx in sss.split(train_data.train_idx):
    # writer = SummaryWriter(log_dir=path+'/fold'+str(fold))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_data.test_idx)

    train_loader = DataLoader(train_data,batch_size=batch_size, 
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_data,batch_size=batch_size, 
                                sampler= valid_sampler, num_workers=num_workers)

    test_loader = DataLoader(test_data,batch_size=batch_size,
                                sampler= test_sampler, num_workers=num_workers)


    model = Network()

    print(model)

    #%%

    epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    #%%

    model.to(device)

    print('Starting to Train...')

    for e in range(1,epochs+1):
        size = len(train_loader)
        print("train size", size)
        model.train()
        train_loss, all_train_preds, all_train_actuals = 0,[],[]
        # with torch.no_grad():
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device).float(), y.to(device).float()
            X = torch.unsqueeze(X, 1).float() 
            pred = model(X)
            pred = pred.squeeze()
            loss = criterion(pred, y)
            train_loss += (loss.item()*X.shape[0])
            all_train_preds.extend(pred.detach().cpu().numpy().tolist())
            all_train_actuals.extend(y.detach().cpu().numpy().tolist())

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_idx)
        print(f"Average train loss: {avg_train_loss}")
        print("Prediction Values: ",all_train_preds )
        print("Actual Values", all_train_actuals )
        # avg_train_loss = train_loss / len(train_loader)
        # print(f"Average train loss: {avg_train_loss}")

        #<!------Valid------->
        # else:
        size = len(valid_loader)
        print("Valid size", size)
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for X,y in valid_loader:
                X, y = X.to(device).float(), y.to(device).float()
                X = torch.unsqueeze(X, 1).float() 
                pred = model(X)
                pred = pred.squeeze() 
                loss = criterion(pred, y)
                valid_loss += ((loss.item())*X.shape[0])

        avg_valid_loss = valid_loss / len(valid_idx)
        scheduler.step(avg_valid_loss)
        # avg_valid_loss = valid_loss / len(valid_loader)
        # print(f"Average valid loss: {avg_valid_loss}")


        #<!------Test-------->
        # size = len(test_loader)
        # print("test size", size)
        # model.eval()
        # test_loss = 0

        # with torch.no_grad():
        #     for X, y in test_loader:
        #         X, y = X.to(device).float(), y.to(device).float()
        #         X = torch.unsqueeze(X, 1).float() 
        #         pred = model(X)
        #         pred = pred.squeeze()  
        #         test_loss += criterion(pred, y).item()

        # avg_test_loss = test_loss / len(test_loader)
        # print(f"Average test loss: {avg_test_loss}")

        
        print("Epoch: {}/{}.. ".format(e, epochs))

        print('Train Loss', train_loss/len(train_idx),e)
        print('Validation Loss', valid_loss/len(valid_idx),e)
        # print('Test Loss', test_loss/len(test_loader),e)
        

    fold+=1
    print('####################################################################')
    print("Done")




