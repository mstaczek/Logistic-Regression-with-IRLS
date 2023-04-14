from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd

def remove_columns(X, column_names):
    return X.drop(column_names, axis=1)

def remove_missing_columns(X, threshold=0.5):
    missing_values = X.isna().sum() / X.shape[0]
    mask = missing_values > threshold
    X = X.drop(missing_values[mask].index, axis=1)
    return X

def remove_missing_rows(X, Y):
    mask = X['Embarked'].isna()
    X = X[~mask]
    Y = Y[~mask]
    return X, Y

def encode_text(X):
    X.loc[:,'Sex'] = X['Sex'].map({'male': -1, 'female': 1})
    X.loc[:,'Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return X

def impute_missing_age(X):
    column_names = X.columns
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)
    return pd.DataFrame(data=X, columns=column_names)

def check_vif(X):
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info.sort_values('VIF', ascending=False)
    return vif_info

def one_hot_encode(X, columns):
    encoder = OneHotEncoder(sparse=False, drop='first')
    for column in columns:
        values = X[column].values.reshape(-1, 1)
        encoded = encoder.fit_transform(values)
        encoded = pd.DataFrame(data=encoded, columns=[f'{column}_{value:.0f}' for value in encoder.categories_[0][1:]])
        X = pd.concat([X, encoded], axis=1)
        X = X.drop(column, axis=1)
    return X

def scale_features(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X
