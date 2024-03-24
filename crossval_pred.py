import numpy as np
import pandas as pd
import argparse
import os

import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from cust_dataset import CustDataset
from network import Network

torch.set_default_dtype(torch.float32)
parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())


#creating directory

directory = args.job_id
parent_directory = '/data/users1/schitikesi1/white_matter/JobOutputs'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models_fold')

if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(model_save_path)


#Loading data
    
torch.manual_seed(52)
num_workers = 4
batch_size = 15
if torch.cuda.device_count() > 1:
    batch_size *= torch.cuda.device_count()
else:
    batch_size = 5
test_size = 0.20

train_data = CustDataset(transform = 
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]),train=True)
test_data = CustDataset(train=False,valid=False)


train_sampler = SubsetRandomSampler(train_data.train_idx)
test_sampler = SubsetRandomSampler(test_data.test_idx)
train_loader = DataLoader(train_data,batch_size=5, 
                                sampler= train_sampler, num_workers=4)

test_loader = DataLoader(test_data,batch_size=5,
                                sampler= test_sampler, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    print("train size", size)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X, y = X.float(), y.float()  # Ensure X is converted to float
        X =torch.unsqueeze(X,1).float()
       

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader)
    print("test size", size)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()
            X = torch.unsqueeze(X, 1).float() 
            pred = model(X)
            pred = pred.squeeze()  
            test_loss += loss_fn(pred, y).item()

    avg_test_loss = test_loss / len(dataloader)
    print(f"Average test loss: {avg_test_loss}")


model = model.double().to(device)
model = model.float()  # Ensure the model is in float

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")


