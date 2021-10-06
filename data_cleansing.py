import numpy as np
import pandas as pd
import re
import time
import os
import inspect

from utils import save_dataset

dataset_original = pd.read_csv('./datasets/ChEMBL_original_dataset.csv', delimiter=';')

# Take out all values that have pChEMBL values
dataset_v1 = dataset_original[dataset_original['pChEMBL Value'].notna()]   # 1236, 45

# Check out the duplicates and take their mean values
dataset_v2 = dataset_v1.groupby('Molecule ChEMBL ID').mean()['Standard Value'].reset_index()

# calculate pChEMBL values
s_value = dataset_v2['Standard Value'].values
p_value = np.around(- np.log10(s_value/(10**(9))), 2)
dataset_v2['Calculated pChEMBL'] = p_value.tolist()


for i in range(0, dataset_v2.shape[0]):
    index = dataset_v2['Molecule ChEMBL ID'][i]
    smile = dataset_v1.loc[dataset_v1['Molecule ChEMBL ID'] == index]['Smiles'].drop_duplicates()
    dataframe = pd.DataFrame(smile)

    if i == 0:
        concat_df = dataframe
    else:
        concat_df = pd.concat([concat_df, dataframe], axis = 0)

concat_df = concat_df.reset_index()

all_structures = pd.concat([dataset_v2, concat_df], axis = 1)

save_dataset(all_structures)
