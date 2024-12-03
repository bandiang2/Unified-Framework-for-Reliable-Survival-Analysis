from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np

from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
# MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
DATA_DIR = Path.joinpath(ROOT_DIR, 'Data')
    
def load(dataset, EPS=1e-8):
    dataset = dataset.lower()
    if dataset == "metabric":
        df = datasets.metabric.read_df()
        df['duration'] += EPS
        
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        
    elif dataset == 'gbsg':
        df = datasets.gbsg.read_df()
        df['duration'] += EPS
        
        cols_standardize = ['x3', 'x4', 'x5', 'x6']
        cols_leave = ['x0', 'x1', 'x2']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        
    elif dataset == 'support':
        data = pd.read_csv('./Data/support2.csv')
        # path = Path.joinpath(DATA_DIR, 'support2.csv')
        # data = pd.read_csv(path)
        
        cols_standardize = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
             'urine', 'adlp', 'adls']
        cols_cats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
        
        x1 = data[cols_standardize]
        x2 = pd.get_dummies(data[cols_cats])
        cols_leave = x2.columns.tolist()
        
        x = np.concatenate([x1, x2], axis=1)
        time = data['d.time'].values
        event = data['death'].values

        x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
        df = pd.DataFrame(data=x, columns=cols_standardize+cols_leave)
        df['duration'] = time
        df['duration'] += EPS
        df['event'] = event

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
    
    elif dataset == 'flchain':
        PATH_DATA = "./Data/flchain.csv"
        df = pd.read_csv(PATH_DATA)
        # path = Path.joinpath(DATA_DIR, 'flchain.csv')
        # data = pd.read_csv(path)
        
        df = df.drop('sample.yr', axis=1)
        df.rename(columns={'futime': 'duration', 'death': 'event'}, inplace=True)
        df['duration'] += EPS
        
        cols_standardize = ['age', 'kappa', 'lambda', 'creatinine']
        cols_leave = ['sex', 'flc.grp', 'mgus']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
    
    else:
        raise NotImplementedError(f"{dataset} not found in the data folder!")
        
    return df, standardize, leave