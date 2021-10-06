import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
import utils
from utils import plot_multi_learning_curves

np.random.seed(42)
dataset = pd.read_csv('./datasets/3_512_x_main.csv')
target = pd.read_csv('./datasets/3_512_y_main.csv')
x_values = dataset.values
y_values = target.values.ravel()

estimator1 = SVR(C=10)
estimator2 = RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_split=4)
estimator3 = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=9)

plot_multi_learning_curves(x_values, y_values, estimator1, estimator2, estimator3,
                           random_seed = 42, testsize = 0.2, mode = 'rmse', autosave = 'y', interval = None)
