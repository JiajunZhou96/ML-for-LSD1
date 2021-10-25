import numpy as np
import pandas as pd
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs


def smiles_dataset(dataset_df = None, smiles_loc = 'Smiles', fp_radius = 3, fp_bits = 1024):

    '''
    Use this function to generate the dataframe of fingerprint
    dataset_df: the input dataset should be a dataframe
    smiles_loc: the column name that consists of SMILES strings
    fp_radius = the radius of Morgan fingerprint
    fp_bits = the number of fingerprint bits of Morgan fingerprint
    '''

    smiles = dataset_df[smiles_loc]
    smiles_list = np.array(smiles).tolist()

    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    mols = [Chem.AddHs(smile) for smile in mols]

    morgans = [AllChem.GetMorganFingerprintAsBitVect(mol, radius = fp_radius,
                nBits= fp_bits, useChirality = True) for mol in mols]
    morgan_bits =  [morgan.ToBitString() for morgan in morgans]

    pattern = re.compile('.{1}')  # find every single digit
    morgan_bits = [','.join(pattern.findall(morgan)) for morgan in morgan_bits]

    fp_list = []
    for bit in morgan_bits:
        single_fp = bit.split(',')   # split the string by commas
        single_fp = [float(fp) for fp in single_fp] # transfer string to float32
        fp_list.append(single_fp)

    fp_df = pd.DataFrame(np.array(fp_list))
    fp_df.columns = fp_df.columns.astype(str)

    # rename the columns
    for i in range(fp_df.columns.shape[0]):
        fp_df.rename(columns = {fp_df.columns[i]:fp_df.columns[i] + "pChEMBL"}, inplace = True)

    return fp_df
