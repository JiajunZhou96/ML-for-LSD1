import json
import numpy as np
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs

from utils import save_dataset, get_parameters
from rdkit_utils import smiles_dataset

dataset = pd.read_csv('./datasets/all_structures.csv')


# change the parameters in .json file
dic = get_parameters(path = './settings/fp_settings.json', print_dict = False)

x = smiles_dataset(dataset_df = dataset, smiles_loc = 'Smiles',
                   fp_radius = dic.get("fp_radius"), fp_bits = dic.get("fp_bits"))

y = dataset['Calculated pChEMBL']

# change file_name to save as different datasets
save_dataset(x, file_name = dic.get("dataset_name"), idx = False)
save_dataset(y, file_name = dic.get("label_name"), idx = False)
