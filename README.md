# ML-for-LSD1

This repository contains codes and data in *Machine Learning Enabled Virtual Screening For Inhibitors of Lysine-Specific Histone Demethylase*.



## Requirements
* Python 3.7
* Numpy 1.19.2
* Pandas 1.1.3
* RDKit 2020.09.1.0
* scikit-learn 0.23.2
* matplotlib 3.3.2





## Dataset for screening
The screening dataset can be found in https://zinc.docking.org/substances/subsets/in-vitro/ . Please add the `in-vitro.csv` file to `./ML-for-LSD1/screening_base/in-vitro_zinc/` directory.


## Script Description


#### Generate machine learning datasets from original pChEMBL dataset.
`data_cleansing.py` and `dataset_construction.py`

#### Hyperparameter Optimization
`optimization.py`

#### Algorithm Fitting
`fitting.py`

#### Part of Analysis
`plot_learning_curves.py` and `tsne.py`

#### Neural Network
`neural network.py`

#### Virtual Screening
`deploy.py` and `deploy2.py`

#### Utils
`utils.py`, `rdkit_utils.py` and `nn_utils.py`
