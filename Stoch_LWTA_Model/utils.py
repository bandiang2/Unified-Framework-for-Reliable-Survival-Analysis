import pandas as pd
import numpy as np
import scipy.stats as st
import yaml
import os
import torch
from sklearn.preprocessing import StandardScaler
import scipy as sp
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize(X_train, X_test, config):
    scaler = StandardScaler()
    continuous_features = config['preprocessing']['continuous_features']
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    return X_train, X_test, scaler


def load_data(dataset_folder):
    dataset = pd.read_csv(os.path.join(dataset_folder, 'data.csv'))
    with open(os.path.join(dataset_folder, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    return dataset, config

def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    return X, y

def convert_to_structured(T, E):
    # dtypes for conversion
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}

    # concat of events and times
    concat = list(zip(E, T))

    # return structured array
    return np.array(concat, dtype=default_dtypes)

def get_prediction(model, x_test, times_ddh, num_samples=5):
    # Perform multiple forward passes
    # print(f"x test: {x_test.shape}")
    # model.eval()
    batch_survs = torch.zeros((num_samples, x_test.shape[0], len(times_ddh)), device=device)
    for i in range(num_samples):
        out_surv = model.predict_survival(x_test, times_ddh.tolist())
        batch_survs[i] = torch.from_numpy(out_surv)
        
    out_survival = batch_survs.mean(0)
    out_survival = out_survival.detach().cpu().numpy()
    # print(f"batch survival: {out_survival.shape}")
    out_risk = 1 - out_survival

    return out_survival, out_risk

def get_prediction_2(model, x_test, times_ddh, samples=5, sample=True):
    out_survival = model.predict_survival(x_test, times_ddh.tolist(), sample=sample, samples=samples)
    # out_survival = out_survival.detach().cpu().numpy()
    out_risk = 1 - out_survival
    return out_survival, out_risk

