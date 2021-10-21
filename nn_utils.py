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
import random

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def result_average_NN(net, x, y, loss, num_epochs, optimizer, testsize = 0.2, random_seeds_start = 2, random_seeds_stop = 62, random_seeds_step = 2, device = None):

    train_scores = []
    test_scores = []
    train_rmses= []
    test_rmses = []


    random_seeds = np.arange(random_seeds_start, random_seeds_stop, random_seeds_step)

    for seeds in random_seeds:

        random.seed(seeds)
        np.random.seed(seeds)
        torch.manual_seed(seeds)
        torch.cuda.manual_seed_all(seeds)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = testsize, random_state = seeds)

        train_x_ts = torch.from_numpy(train_x).float()
        test_x_ts = torch.from_numpy(test_x).float()
        train_y_ts = torch.squeeze(torch.from_numpy(train_y).float())
        test_y_ts = torch.squeeze(torch.from_numpy(test_y).float())
        train_x = train_x_ts.to(device)
        test_x = test_x_ts.to(device)
        train_y = train_y_ts.to(device)
        test_y = test_y_ts.to(device)

        for epoch in range(num_epochs):

            pred = torch.squeeze(net(train_x))
            l = loss(pred, train_y)

            train_loss = l.item()

            if epoch == 0 or (epoch + 1) % 1000 == 0:

                test_pred = torch.squeeze(net(test_x))
                test_loss = loss(test_pred, test_y)
                test_loss += test_loss.item()

                # r2 score
                train_score = functional.r2score(pred, train_y)
                test_score = functional.r2score(test_pred, test_y)

                # rmse
                train_rmse = torch.sqrt(((pred - train_y)*(pred - train_y)).sum()/ len(train_y)).data
                test_rmse = torch.sqrt(((test_pred - test_y)*(test_pred - test_y)/ len(test_y)).sum()).data


            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        train_score = functional.r2score(pred, train_y)
        test_score = functional.r2score(test_pred, test_y)
        train_rmse = torch.sqrt(((pred - train_y)*(pred - train_y)).sum()/ len(train_y)).data
        test_rmse = torch.sqrt(((test_pred - test_y)*(test_pred - test_y)/ len(test_y)).sum()).data


        train_scores.append(train_score.cpu().detach().numpy())
        test_scores.append(test_score.cpu().detach().numpy())
        train_rmses.append(train_rmse.cpu().detach().numpy())
        test_rmses.append(test_rmse.cpu().detach().numpy())

        net.apply(weight_reset)

    avg_train_score = np.average(train_scores)
    avg_score = np.average(test_scores)
    avg_train_rmse = np.average(train_rmses)
    avg_test_rmse = np.average(test_rmses)

    train_score_std = np.std(train_scores)
    test_score_std = np.std(test_scores)
    train_rmse_std = np.std(train_rmses)
    test_rmse_std = np.std(test_rmses)


    print('The average train score of all random states:{:.3f}'.format(avg_train_score))
    print('The average test score of all random states:{:.3f}'.format(avg_score))
    print('The average train rmse of all random states:{:.3f}'.format(avg_train_rmse))
    print('The average test rmse of all random states:{:.3f}'.format(avg_test_rmse))
    print('---------------------------------')
    print('The train score std: {:.4f}'.format(train_score_std))
    print('The test score std: {:.4f}'.format(test_score_std))
    print('The train rmse std: {:.4f}'.format(train_rmse_std))
    print('The test rmse std: {:.4f}'.format(test_rmse_std))
