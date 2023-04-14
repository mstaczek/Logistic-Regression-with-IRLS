import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from LogisticRegressionIRLS import LR
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from data_preprocessing_breast_cancer import *
from data_preprocessing_titanic import *


class LRWrapper(LR):
    def __init__(self, *args, all_interactions=False, **kwargs):
        self.all_interactions = all_interactions
        super().__init__(*args, **kwargs)

    def fit(self, X, Y):
        X = pd.DataFrame(X)
        Y = Y.reshape(-1, 1)
        if self.all_interactions:
            interactions = [[i, j] for i in range(X.shape[1]) for j in range(i+1, X.shape[1])]
            super().fit(X, Y, interactions=interactions)
        else:
            super().fit(X, Y)
        return self
    
    def predict(self, X):
        X = pd.DataFrame(X)
        Y = super().predict(X)
        return Y

def metric_bootstrap(X, Y, model_class, n=10):
    model = model_class()
    error = 0
    lenght_of_y = 0
    for i in range(n):
        sample_ids = resample(range(len(X)), replace=True, n_samples=len(X))

        test_ids = [i for i in range(len(X)) if i not in sample_ids]
        if len(sample_ids) == len(test_ids):
            i -= 1
            continue
        lenght_of_y += len(test_ids)
        train_ids = [i for i in range(len(X)) if i in sample_ids]
        model.fit(X[train_ids], Y[train_ids])
        y_pred = model.predict(X[test_ids])
        for i in range(len(y_pred)):
            if Y[test_ids[i]] != y_pred[i]:
                error += 1
    error_bootstrap = error / lenght_of_y
    accuracy_bootstrap = 1 - error_bootstrap
    return accuracy_bootstrap

def train_model(model_class, X_train, X_test, Y_train, Y_test, metrics, model_name='model'):
    model = model_class()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return {'model':model_name} |\
            {metric: metrics[metric](Y_test, Y_pred) for metric in metrics if metric != 'bootstrap'} |\
            {f'bootstrap accuracy': metrics['bootstrap'](X_train, Y_train, model_class)}

def train_models(models, X, Y, metrics, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    results = []
    for model_name, model_class in models.items():
        results.append(train_model(model_class, X_train, X_test, Y_train, Y_test, metrics, model_name=model_name))
    df = pd.DataFrame(results)
    return df

def generate_plot(data, repetitions, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=data, ax=ax, x='metric', y='value', hue='model', width=0.5)


    ax.set_title(title)
    ax.set_xlabel('Metric (on test set, except for bootstrap))')
    ax.set_ylabel(f'Metric value ({repetitions} repetitions)')
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Model')
    fig.savefig(filename, dpi=300)


def experiment_titanic(models, metrics, repetitions=10):
    # ## Dataset - Titanic
    # Source: [Kaggle](https://www.kaggle.com/c/titanic/data)

    print('---- Experiment: Titanic - beginning ----')

    if 'titanic' not in os.listdir():
        print('Place titanic dataset at "titanic/train.csv"')
        print('Titanic not found, skipping experiment.')
        return
    df = pd.read_csv('titanic/train.csv')

    # PREPROCESSING START
    print('Before processing: rows: ', df.shape[0], 'columns: ', df.shape[1])
    X, Y = df.drop('Survived', axis=1), df['Survived'].to_numpy()
    X = remove_columns(X, ['Name', 'Ticket', 'PassengerId'])
    X = remove_missing_columns(X) # will remove the Cabin column
    X, Y = remove_missing_rows(X, Y) # remove 2 rows in Embarked column
    X = encode_text(X) # sex to -1, 1, embarked one hot encoding
    X = impute_missing_age(X) # imputate missing values in Age
    
    print('VIF colinearity check for titanic dataset:')
    print(check_vif(X))
    print('No colinearity present.')

    X = one_hot_encode(X, ['Embarked', 'Pclass'])
    X = scale_features(X)
    print('After processing: rows: ', X.shape[0], 'columns: ', X.shape[1])
    # PREPROCESSING END

    test_size = 0.2

    df_list = []
    for i in tqdm(range(repetitions)):
        df_list.append(train_models(models, X, Y, metrics, test_size))
    df = pd.concat(df_list, ignore_index=True)
    df = df.melt(id_vars=['model'], var_name='metric', value_name='value')

    generate_plot(data=df, repetitions=repetitions, title='Models compared for the Titanic dataset',
                  filename='models_comparison_2_titanic.png')

    print('---- Experiment: Titanic - finished ----')


def experiment_breast_cancer(models, metrics, repetitions=10):
    # ## Dataset - Breast Cancer
    # Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

    print('---- Experiment: Breast Cancer - beginning ----')
    X, Y = load_breast_cancer(return_X_y=True)
    df = pd.DataFrame(data=np.hstack((X, Y.reshape(-1, 1))), columns=list(load_breast_cancer().feature_names) + ['target'])
    
    print(f"In the raw data, there are {X.shape[0]} observations and {X.shape[1]} features.")
    X, columns_selection = remove_colinear(X, correlation_threshold=0.9)
    X = scale_features(X)
    print(f"In the prepared data, there are {X.shape[0]} observations and {X.shape[1]} features" +\
          f" (removed {len(columns_selection) - X.shape[1]} features).")

    test_size = 0.6 # it is this high to see differences in the results

    df_list = []
    for _ in tqdm(range(repetitions)):
        df_list.append(train_models(models, X, Y, metrics, test_size))
    df = pd.concat(df_list, ignore_index=True)
    df = df.melt(id_vars=['model'], var_name='metric', value_name='value')

    generate_plot(data=df, repetitions=repetitions, title='Models compared for the Breast Cancer dataset',
                   filename='models_comparison_1_breast_cancer.png')

    print('---- Experiment: Breast Cancer - finished ----')


def run_experiments():

    metrics = {'precision': partial(precision_score, zero_division=0),
                'recall': recall_score,
                'accuracy': accuracy_score,
                'f1': f1_score,
                'bootstrap': metric_bootstrap}

    models = {'LDA': LinearDiscriminantAnalysis,
            'QDA': QuadraticDiscriminantAnalysis,
            'Logistic ': LogisticRegression,
            'Logistic IRLS': partial(LRWrapper, maximisation_minorisation=True),
            'Logistic IRLS with interactions': partial(LRWrapper, maximisation_minorisation=True, all_interactions=True),
            'KNN': KNeighborsClassifier}
    
    experiment_titanic(models, metrics, repetitions=10)
    experiment_breast_cancer(models, metrics, repetitions=10)

if __name__ == '__main__':
    run_experiments()