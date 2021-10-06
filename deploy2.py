import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os
from sklearn.svm import SVR
import joblib

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_utils import smiles_dataset
from utils import save_dataset


model_load = joblib.load('./models/model.pkl')
database = pd.read_csv('./screening_base/in-vitro_zinc/in-vitro.csv')
screen_database = pd.read_csv('./datasets/screen_results/in-vitro_zinc/in-vitro_bits.csv')

screen_result = model_load.predict(screen_database)
screen_result_fp = pd.DataFrame({'Predictive Results': screen_result})
database_result = pd.concat([database, screen_result_fp], axis = 1)

threshold_7 = database_result[database_result['Predictive Results'] > 7]

original_dataset = pd.read_csv('./datasets/all_structures.csv')
de_threshold_7 = threshold_7
for smile in original_dataset['Smiles']:
    for new_structure in threshold_7['smiles']:
        if smile == new_structure:
            index = threshold_7[threshold_7['smiles'] == smile].index[0] # 加上[0] 才能将这个从 奇怪的格式转换为 int 数字
            print('overlap found at position: {:01d}'.format(index))
            de_threshold_7 = de_threshold_7.drop(index = index, axis = 0)
        else:
            pass

save_dataset(threshold_7, path = './datasets/screen_results/in-vitro_zinc/', file_name = 'threshold_7', idx = False)
save_dataset(de_threshold_7, path = './datasets/screen_results/in-vitro_zinc/', file_name = 'de_threshold_7', idx = False)
