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
from utils import save_dataset, get_parameters

model_load = joblib.load('./models/model.pkl')
database = pd.read_csv('./screening_base/in-vitro_zinc/in-vitro.csv')

dic = get_parameters(path = './settings/fp_settings.json', print_dict = False)
database_fp = smiles_dataset(dataset_df = database, smiles_loc = 'smiles',
                   fp_radius = dic.get("fp_radius"), fp_bits = dic.get("fp_bits"))

save_dataset(database_fp, path = './datasets/screen_results/in-vitro_zinc/', file_name = 'in-vitro_bits', idx = False)
