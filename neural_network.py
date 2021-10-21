# Use the script under google colab/GPU environment

import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pytorch_lightning.metrics import functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

from nn_utils import weight_reset, result_average_NN

print('GPU availability:', torch.cuda.is_available())

dataset = pd.read_csv('./datasets/3_512_x_main.csv')
target = pd.read_csv('./datasets/3_512_y_main.csv')

class predict_model(nn.Module):
    def __init__(self, x_size, hidden1_size, hidden2_size, y_size):
        super(predict_model, self).__init__()
        self.hidden1 = nn.Linear(x_size, hidden1_size)
        self.batch1 = nn.BatchNorm1d(hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.batch2 = nn.BatchNorm1d(hidden2_size)
        self.predict = nn.Linear(hidden2_size, y_size)
    def forward(self, input):
        result = self.hidden1(input)
        result = self.batch1(result)
        result = F.leaky_relu(result)
        result = self.hidden2(result)
        result = self.batch2(result)
        result = F.leaky_relu(result)
        result = self.predict(result)

        return result

net = predict_model(dataset.shape[1], 256, 64, 1)
loss_func = nn.MSELoss(reduction = 'sum')
optimizer = optim.Adam(net.parameters(), lr = 0.02, eps = 1e-08, weight_decay = 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
loss_func = loss_func.to(device)

net.apply(weight_reset)
result_average_NN(net, dataset.values, target.values, loss = loss_func, 
                  num_epochs = 10000, optimizer = optimizer, testsize = 0.2, random_seeds_start = 2, random_seeds_stop = 62, random_seeds_step = 2, device = device)
