# Logistic-Regression-with-IRLS
Project for studies course Andvanced Machine Learning.

--------

## File structure

There are 3 categories of files here: simulation experiments, experiments on real data, and results (plots).

**Simulation experiments:**
- `simulation_experiments.ipynb` - Jupyter notebook showing the experiments on simulated data sets,
- `simulation_functions.py` - functions used in this notebook.

**Real data experiments:**
- `real_data_experiments.ipynb` -  Jupyter notebook showing the experiments on 2 data sets,
- `real_data_experiments.py` - a Python script that conducts the same experiments as the notebook,
- `data_preprocessing_breast_cancer.py` and `data_preprocessing_titanic.py` - contain functions used in the script; they are also present in the Jupyter notebook,

**Plots:**
- `training_iter_*.png` - decision boundary during first few steps in training on simulated data,
- `models_comparison_*.png` - plots with accuracies of models trained on real datasets.

## Datasets used for experiments

1. [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
