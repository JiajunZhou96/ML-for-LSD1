import json
import csv
import inspect
import os
import re
import numpy as np
import pandas as pd
import time

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt



def retrieve_df_name(dataframe):
    '''
    Use this function to retrieve the name of the dataframe
    '''
    for fi in reversed(inspect.stack()):
        df_names = [df_name for df_name, df_val in fi.frame.f_locals.items() if df_val is dataframe]
        if len(df_names) > 0:
            return df_names[0]

def save_dataset(dataframe, path = None, file_name = None, idx = False):
    '''
    Use this function to save the dataframe
    dataframe: DataFrame
    path: folder path in which the dataset stores, a string with ''
    file_name: please enter a string with ''
    idx: to control the creation of index or not
    '''
    if path is None:
        path = os.path.join(os.getcwd(), 'datasets')
    else:
        path = path
    print('Current path is:', path)

    if os.path.exists(path) == True:
        pass
        print('Path already existed.')
    else:
        os.mkdir(path)
        print('Path created.')

    if file_name is None:

        dataframe.to_csv(path + '/' + retrieve_df_name(dataframe)+ '.csv', index = idx)
    else:
        dataframe.to_csv(path + '/' + file_name + '.csv', index = idx)

    print('Dataset saved successfully.')

def get_parameters(path = None, print_dict = False):

    if path is None:
        raise ValueError('No path entered.')
    else:
        path = path
    print('json file path is:', path)

    f = open(path, 'r')
    line = f.read()
    dic = json.loads(line)

    if print_dict == False:
        pass
    elif print_dict == True:
        print(dic)

    return dic

def pipeline_optim(x_df, y_df, random_seeds = 42, testsize = 0.2, mode = 'rf', scoring = None):

    '''
    x_df: the dataframe of the dataset, training set + cross validation set
    y_df: the dataframe of the target, training set + cross validation set

    scoring: 'None' to use r^2, 'mse_func' to use MSE for performance evaluation
    '''

    start = time.time()
    np.random.seed(random_seeds)

    # dataset x and y
    if mode in ['knn','ridge','lasso','elastic','dt']:
    #if mode == 'knn' or mode =='ridge' or mode == ...

        x_data = x_df.values
        y_data = y_df.values

    elif mode in ['gradientboosting','adaboost','extratrees','rf', 'dt' ,'svr']:

        x_data = x_df.values
        y_data = y_df.values
        y_data = y_data.flatten()

    else:
        raise ValueError('Mode not found')


    dataframes = []
    best_estimators = []
    estimators = []
    param_grid = {}

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = testsize, random_state = random_seeds)


    if mode == 'knn':
        regr = KNeighborsRegressor()
        param_grid = {'n_neighbors': [2,3,4,5,6,7,8,9,10],
                      'weights':['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [20, 30, 40],
                      'p': [1,2]}
    if mode == 'ridge':
        regr = Ridge()
        params_grid = {'alpha': [0.001,0.01,0.1,1,10],
                       'solver': ['auto'],
                       'max_iter': [100000]}
    if mode == 'lasso':
        regr = Lasso()
        param_grid = {'alpha': [0.001,0.01,0.1,1,10],
                      'selection':['cyclic', 'random'],
                      'max_iter': [100000]}
    if mode == 'elastic':
        regr = ElasticNet()
        param_grid = {'alpha':[0.001,0.01,0.1,1,10],
                      'l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      'max_iter': [100000]}
    if mode == 'gradientboosting':
        regr = GradientBoostingRegressor()
        param_grid = {'learning_rate': [0.001,0.01,0.1,1],
                         'min_samples_split': [2,3,4,5,6,7,8,9],
                         'loss': ['ls', 'lad', 'huber', 'quantile'],
                         'criterion':['mse', 'friedman_mse']}
    if mode == 'adaboost':
        regr = AdaBoostRegressor()
        param_grid = {'learning_rate': [0.001,0.01,0.1,1],
                         'loss': ['linear', 'square', 'exponential']}
    if mode == 'extratrees':
        regr = ExtraTreesRegressor()
        param_frid = {'bootstrap': [True, False],
                      'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'rf':
            regr = RandomForestRegressor()
            param_grid = {'bootstrap': [True, False],
                         'max_features': ['auto','log2','sqrt'],
                         'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'dt':
        regr = DecisionTreeRegressor()
        param_grid = {'criterion': ['mse', 'friedman_mse'],
                      'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'svr':
        regr = SVR()
        param_grid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                      'gamma': ['scale', 'auto'],
                      'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100]}


    regr_grid = GridSearchCV(regr, param_grid, scoring = None)
    regr_grid.fit(x_train, y_train)

    best_estimator = regr_grid.best_estimator_


    print('The best hyperparameters:', best_estimator)
    print('Time for optimization: %f seconds' %(time.time()-start), flush=True)
    print('*********************************')

    return best_estimator

def fit_result(x, y, estimator, random_seeds_start = 2, random_seeds_stop = 62, random_seeds_step = 2, testsize = 0.2):


    start = time.time()

    test_scores = []
    train_scores = []
    train_mses = []
    mses = []
    train_rmses = []
    rmses = []

    random_seeds = np.arange(random_seeds_start, random_seeds_stop, random_seeds_step)

    for seeds in random_seeds:

        np.random.seed(seeds)   # np.arange(2,62,2) is random seed from 2 to 60
        x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = testsize, random_state = seeds)

        regr = estimator
        regr.fit(x_train, y_train)
        train_pred = regr.predict(x_train)
        pred = regr.predict(x_test)


        train_score = regr.score(x_train,y_train)
        test_score = regr.score(x_test,y_test)
        #print('random seed',seeds,'train score: {:.3f}'.format(train_score))
        #print('random seed',seeds,'test score:{:.3f}'.format(test_score))

        train_mse = sum((x - y)*(x - y) for x, y in zip(train_pred, y_train))/ len(x_train)
        train_rmse = np.sqrt(train_mse)
        #print('random seed',seeds,'Train RMSE:{:.3f}'.format(train_rmse))
        mse = sum((x - y)*(x - y) for x, y in zip(pred, y_test))/ len(x_test)
        rmse = np.sqrt(mse)
        #print('random seed',seeds,'Test RMSE:{:.3f}'.format(rmse))

        #print('--------------------------------------')

        train_scores.append(train_score)
        train_mses.append(train_mse)
        train_rmses.append(train_rmse)
        test_scores.append(test_score)
        mses.append(mse)
        rmses.append(rmse)

    # averages
    avg_train_score = np.average(train_scores)
    avg_score = np.average(test_scores)
    avg_train_rmse = np.average(train_rmses)
    avg_rmse = np.average(rmses)

    # standard deviations
    train_score_std = np.std(train_scores)
    test_score_std = np.std(test_scores)
    train_rmse_std = np.std(train_rmses)
    rmse_std = np.std(rmses)

    print('The average train score of all random states:{:.3f}'.format(avg_train_score))
    print('The average test score of all random states:{:.3f}'.format(avg_score))
    print('The average train rmse of all random states:{:.3f}'.format(avg_train_rmse))
    print('The average test rmse of all random states:{:.3f}'.format(avg_rmse))

    print('---------------------------------')
    print('The train score std: {:.4f}'.format(train_score_std))
    print('The test score std: {:.4f}'.format(test_score_std))
    print('The train rmse std: {:.4f}'.format(train_rmse_std))
    print('The test rmse std: {:.4f}'.format(rmse_std))

    print('---------------------------------')
    print('Time for fitting: %f seconds' %(time.time()-start), flush=True)
    print('*********************************')

def extract_data_for_plot(y, interval = None):
    '''
    work with utils.plot_multi_learning_curves
    '''
    if isinstance(interval, int):

        index1 = list(np.arange(interval, len(y), interval))
        last = [len(y) - 1]
        index1.insert(0, 0)
        index1.extend(last)
        index2 = list(set(index1))
        index2.sort(key = index1.index)
        y = np.array(y)
        y_plot = list(y[index2])
        x_plot = [i + 1 for i in index2]

    elif interval is None:

        index2 = list(np.arange(0, len(y)))
        x_plot = [i + 1 for i in index2]
        y_plot = y

    else:
        raise ValurError('interval must be an integer or None')

    return x_plot, y_plot

def plot_multi_learning_curves(x, y, estimator1, estimator2, estimator3, random_seed = 42, testsize = 0.2, mode = 'r2', autosave = 'n', interval = 1, path = None):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testsize, random_state = random_seed)

    train_errors1 = []
    test_errors1 = []
    train_errors2 = []
    test_errors2 = []
    train_errors3 = []
    test_errors3 = []

    if mode == 'rmse':

        for m in range(1, len(x_train)):

            estimator1.fit(x_train[: m], y_train[: m])

            train_pred = estimator1.predict(x_train[: m])
            test_pred = estimator1.predict(x_test)

            train_errors1.append(mean_squared_error(y_train[: m], train_pred))
            test_errors1.append(mean_squared_error(y_test, test_pred))

            estimator2.fit(x_train[: m], y_train[: m])

            train_pred = estimator2.predict(x_train[: m])
            test_pred = estimator2.predict(x_test)

            train_errors2.append(mean_squared_error(y_train[: m], train_pred))
            test_errors2.append(mean_squared_error(y_test, test_pred))

            estimator3.fit(x_train[: m], y_train[: m])

            train_pred = estimator3.predict(x_train[: m])
            test_pred = estimator3.predict(x_test)

            train_errors3.append(mean_squared_error(y_train[: m], train_pred))
            test_errors3.append(mean_squared_error(y_test, test_pred))

        train_errors1_x, train_errors1 = extract_data_for_plot(train_errors1, interval = interval)
        test_errors1_x, test_errors1 = extract_data_for_plot(test_errors1, interval = interval)
        train_errors2_x, train_errors2 = extract_data_for_plot(train_errors2, interval = interval)
        test_errors2_x, test_errors2 = extract_data_for_plot(test_errors2, interval = interval)
        train_errors3_x, train_errors3 = extract_data_for_plot(train_errors3, interval = interval)
        test_errors3_x, test_errors3 = extract_data_for_plot(test_errors3, interval = interval)


        plt.figure(figsize=(16, 8))
        plt.xticks(size = 22)
        plt.yticks(size = 22)
        plt.xlabel('Number of Examples', fontproperties = 'Times New Roman', fontsize = 24)
        plt.ylabel('Root Mean Square Error (RMSE)', fontproperties = 'Times New Roman', fontsize = 24)


        plt.plot(train_errors1_x, train_errors1, color = 'blue',linestyle = '--', linewidth = 1, label = 'SVR Train R\u00b2') #label = 'SVR Train R\u00b2'
        plt.plot(test_errors1_x, test_errors1, color = 'blue', linestyle = '-', linewidth = 1, label = 'SVR Test R\u00b2')  # label = 'SVR Test R\u00b2'
        plt.plot(train_errors2_x, train_errors2, color = 'red', linestyle = '--', linewidth = 1, label = 'RF Train R\u00b2') # label = 'RF Train R\u00b2'
        plt.plot(test_errors2_x, test_errors2, color = 'red', linestyle = '-',linewidth = 1, label = 'RF Test R\u00b2') # label = 'RF Test R\u00b2'
        plt.plot(train_errors3_x, train_errors3, color = 'grey', linestyle = '--', linewidth = 1, label = 'DT Train R\u00b2') # label = 'DT Train R\u00b2'
        plt.plot(test_errors3_x, test_errors3, color = 'grey', linestyle = '-', linewidth = 1, label = 'DT Test R\u00b2') # label = 'DT Train R\u00b2'

    elif mode == 'r2':

        for m in range(2, len(x_train)):   # can not calculate the r^2 value below 2
            estimator1.fit(x_train[: m], y_train[: m])

            train_error1 = estimator1.score(x_train[: m], y_train[: m])
            test_error1 = estimator1.score(x_test, y_test)

            train_errors1.append(train_error1)
            test_errors1.append(test_error1)

            estimator2.fit(x_train[: m], y_train[: m])

            train_error2 = estimator2.score(x_train[: m], y_train[: m])
            test_error2 = estimator2.score(x_test, y_test)

            train_errors2.append(train_error2)
            test_errors2.append(test_error2)

            estimator3.fit(x_train[: m], y_train[: m])

            train_error3 = estimator3.score(x_train[: m], y_train[: m])
            test_error3 = estimator3.score(x_test, y_test)

            train_errors3.append(train_error3)
            test_errors3.append(test_error3)

        train_errors1_x, train_errors1 = extract_data_for_plot(train_errors1, interval = interval)
        test_errors1_x, test_errors1 = extract_data_for_plot(test_errors1, interval = interval)
        train_errors2_x, train_errors2 = extract_data_for_plot(train_errors2, interval = interval)
        test_errors2_x, test_errors2 = extract_data_for_plot(test_errors2, interval = interval)
        train_errors3_x, train_errors3 = extract_data_for_plot(train_errors3, interval = interval)
        test_errors3_x, test_errors3 = extract_data_for_plot(test_errors3, interval = interval)

        plt.figure(figsize=(16, 8))
        plt.xticks(size = 22)
        plt.yticks(size = 22)

        plt.xlabel('Number of Examples', fontproperties = 'Times New Roman', fontsize = 24)
        plt.ylabel('Coefficient of Determination (R\u00b2)', fontproperties = 'Times New Roman', fontsize = 24)

        plt.plot(train_errors1_x, train_errors1, color = 'blue',linestyle = '--', linewidth = 1, label = 'SVR Train R\u00b2') #label = 'SVR Train R\u00b2'
        plt.plot(test_errors1_x, test_errors1, color = 'blue', linestyle = '-', linewidth = 1, label = 'SVR Test R\u00b2')  # label = 'SVR Test R\u00b2'
        plt.plot(train_errors2_x, train_errors2, color = 'red', linestyle = '--', linewidth = 1, label = 'RF Train R\u00b2') # label = 'RF Train R\u00b2'
        plt.plot(test_errors2_x, test_errors2, color = 'red', linestyle = '-',linewidth = 1, label = 'RF Test R\u00b2') # label = 'RF Test R\u00b2'
        plt.plot(train_errors3_x, train_errors3, color = 'grey', linestyle = '--', linewidth = 1, label = 'DT Train R\u00b2') # label = 'DT Train R\u00b2'
        plt.plot(test_errors3_x, test_errors3, color = 'grey', linestyle = '-', linewidth = 1, label = 'DT Test R\u00b2') # label = 'DT Train R\u00b2'

    else:
        raise ValurError('mode type rather \'rmse\' or \'r2\'')

    # save figures
    if autosave == 'y':

        if path is None:
            path = os.path.join(os.getcwd(), 'figures')
        else:
            path = path
        print('Current path is:', path)

        if os.path.exists(path) == True:
            pass
            print('Path already existed.')
        else:
            os.mkdir(path)
            print('Path created.')

        plt.savefig(path + '/' + str(mode) + '_compare_learning_curve.png')
        plt.show()
        print('Figure saved successfully.')

    elif autosave == 'n':
        pass
    else:
        raise ValurError('autosave rather \'n\' or \'y\'')
