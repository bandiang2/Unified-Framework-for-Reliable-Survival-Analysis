import pandas as pd
import numpy as np
import torch
import sys, os
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc, integrated_brier_score
from BNN_Sparse_Monotonic import load_datasets, load_tcga_datasets, utils, BayesianSparseNeuralSurvival

def run_experiment(x, y, params, config, horizons=[0.25, 0.5, 0.75], cv=5, verbose=True):
    # Extract parameters
    layers = [params['hidden_dim']] * params['num_layers']
    U, lr, batch_size, num_epochs = params['U'], params['lr'], params['batch_size'], params['n_epochs']
    beta, alpha, num_samples, seed, temperature = params['beta'], params['alpha'], params['num_samples'], params['seed'], params['temperature']
    
    set_seed(seed)
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
    t, e = y[:, 0], y[:, 1]
    continuous_features = config['preprocessing']['continuous_features']
    times = np.quantile(t[e != 0], horizons)
    float_index = pd.Float64Index(times.tolist(), name='level_1')
    
    # Initialize result containers
    metrics = initialize_metrics(cv)
    
    for idx, (train_index, test_index) in enumerate(kf.split(x, e)):
        train_index_, dev_index, val_index = split_indices(train_index)
        x_train, x_val, x_dev, x_test, scaler = preprocess_data(x, train_index_, val_index, dev_index, test_index, continuous_features, config)
        t_train, t_dev, t_val, e_train, e_dev, e_val = split_targets(t, e, train_index_, val_index, dev_index)

        model = train_model(x_train, t_train, e_train, layers, U, temperature, beta, alpha, num_epochs, batch_size, lr, seed, x_dev, t_dev, e_dev, times)
        
        et_train, et_test, x_test = prepare_eval_data(t, e, train_index, test_index, x_test, t_train)
        evaluate_model(model, x_train, x_dev, x_val, x_test, et_train, et_test, t_val, e_val, t_train, times, metrics, idx, num_samples, float_index)
        
    if verbose:
        print_metrics(metrics, cv)
    return metrics

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_metrics(cv):
    return {
        'ciss': {i: np.zeros(cv) for i in range(3)},
        'brss': {i: np.zeros(cv) for i in range(3)},
        'roc_aucs': {i: np.zeros(cv) for i in range(3)},
        'nlls': np.zeros(cv),
        'oCis': np.zeros(cv),
        'oBrs': np.zeros(cv),
        'oNblls': np.zeros(cv)
    }

def split_indices(train_index, proportion=0.1):
    ten_percent = int(proportion * len(train_index))
    dev_index = train_index[:ten_percent]
    val_index = train_index[ten_percent:2 * ten_percent]
    train_index_ = train_index[2 * ten_percent:]
    return train_index_, dev_index, val_index

def preprocess_data(x, train_index_, val_index, dev_index, test_index, continuous_features, config):
    x_train, x_test, scaler = utils.standardize(x.iloc[train_index_], x.iloc[test_index], config)
    x_val = x.iloc[val_index]
    x_val[continuous_features] = scaler.transform(x_val[continuous_features])
    x_dev = x.iloc[dev_index]
    x_dev[continuous_features] = scaler.transform(x_dev[continuous_features])
    return x_train.values, x_val.values, x_dev.values, x_test.values, scaler

def split_targets(t, e, train_index_, val_index, dev_index):
    return t[train_index_], t[dev_index], t[val_index], e[train_index_], e[dev_index], e[val_index]

def train_model(x_train, t_train, e_train, layers, U, temperature, beta, alpha, num_epochs, batch_size, lr, seed, x_dev, t_dev, e_dev, times):
    minmax = lambda x: (x / t_train.max()) + 1e-4
    t_train_ddh, t_dev_ddh = minmax(t_train), minmax(t_dev)
    model = BayesianSparseNeuralSurvival(layers=layers, U=U, temperature=temperature, beta=beta, alpha=alpha, seed=seed)
    model.fit(x_train, t_train_ddh, e_train, n_iter=num_epochs, bs=batch_size, lr=lr, val_data=(x_dev, t_dev_ddh, e_dev), random_state=seed)
    return model

def prepare_eval_data(t, e, train_index, test_index, x_test, t_train):
    et_train = np.array([(e[train_index][i] == 1, t[train_index][i]) for i in range(len(e[train_index]))], dtype=[('e', bool), ('t', float)])
    selection = t[test_index] < t_train.max()
    et_test = np.array([(e[test_index][i] == 1, t[test_index][i]) for i in range(len(e[test_index]))], dtype=[('e', bool), ('t', float)])[selection]
    x_test = x_test[selection]
    return et_train, et_test, x_test

def evaluate_model(model, x_train, x_dev, x_val, x_test, et_train, et_test, t_val, e_val, t_train, times, metrics, idx, num_samples, float_index):
    times_ddh = (np.array(times) / t_train.max()) + 1e-4
    t_val_ddh = (t_val / t_train.max()) + 1e-4
    out_survival_train, out_risk_train = utils.get_prediction(model, np.concatenate([x_train, x_dev, x_val], axis=0), times_ddh, num_samples)
    out_survival, out_risk = utils.get_prediction(model, x_test.astype(float), times_ddh, num_samples)

    out_surv_df_train = pd.DataFrame(out_survival_train, columns=times)
    out_surv_df = pd.DataFrame(out_survival, columns=times)
    out_surv_df_train.columns = pd.MultiIndex.from_frame(pd.DataFrame(index=out_surv_df_train.columns).reset_index().astype(float))
    out_surv_df.columns = pd.MultiIndex.from_frame(pd.DataFrame(index=out_surv_df.columns).reset_index().astype(float))
    out_surv_df_train = out_surv_df_train.T
    out_surv_df_train.index = float_index
    out_surv_df = out_surv_df.T
    out_surv_df.index = float_index
    
    km = EvalSurv(out_surv_df_train, et_train['t'], et_train['e'] == 1, censor_surv='km')
    test_eval = EvalSurv(out_surv_df, et_test['t'], et_test['e'] == 1, censor_surv=km)
    
    update_metrics(metrics, et_train, et_test, out_risk, out_survival, times, test_eval, idx)
    nll = model.compute_nll(x_val, t_val_ddh, e_val)
    metrics['nlls'][idx] = nll

def update_metrics(metrics, et_train, et_test, out_risk, out_survival, times, test_eval, idx):
    for i, _ in enumerate(times):
        metrics['ciss'][i][idx] = concordance_index_ipcw(et_train, et_test, out_risk[:, i], times.tolist()[i])[0]
        metrics['roc_aucs'][i][idx] = cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times.tolist()[i])[0]
        metrics['brss'][i][idx] = brier_score(et_train, et_test, out_survival, times.tolist())[1][i]
    metrics['oBrs'][idx] = test_eval.integrated_brier_score(times)
    metrics['oCis'][idx] = test_eval.concordance_td()
    metrics['oNblls'][idx] = test_eval.integrated_nbll(times)

def print_metrics(metrics, cv):
    print(f"Overall CIS: {np.mean(metrics['oCis']):.3f} ({np.std(metrics['oCis']):.3f})")
    print(f"Overall BRS: {np.mean(metrics['oBrs']):.3f} ({np.std(metrics['oBrs']):.3f})")
    print(f"Overall Nbll: {np.mean(metrics['oNblls']):.3f} ({np.std(metrics['oNblls']):.3f})")
