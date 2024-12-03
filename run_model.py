import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('./')
sys.path.append('./BNN_ISD')
sys.path.append('./DeepSurvivalMachines/')
sys.path.append('./baysurv/')

import numpy as np
import pandas as pd
import random, math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn_pandas import DataFrameMapper
from torch.utils.data import TensorDataset, DataLoader, Dataset

from lifelines.statistics import logrank_test
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc, integrated_brier_score
from SurvivalEVAL.Evaluator import LifelinesEvaluator, SurvivalEvaluator
from lifelines import KaplanMeierFitter

import gc, optuna
from tqdm import tqdm
from time import time

from Stoch_LWTA_Model import load_datasets, load_tcga_datasets, utils, experiment_utils
from Stoch_LWTA_Model import BayesianSparseNeuralSurvival
from datasets import make_data
from utility.training import get_data_loader, scale_data, split_time_event, make_stratified_split
from utility.survival import calculate_event_times, make_time_bins, calculate_percentiles, convert_to_structured

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

# Load data
def load(dataset_name = "METABRIC", ctype='All', seed=seed, noisy=False):
    dl = get_data_loader(dataset_name, ctype=ctype).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    # df.to_csv(f"Data/Seer.csv", index=False)

    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=seed)

    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    if dataset_name != "TCGA":
      X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)

    # Convert to array
    X_train = np.array(X_train).astype(float)
    X_test = np.array(X_test).astype(float)
    X_valid = np.array(X_valid).astype(float)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_test, e_test = split_time_event(y_test)
    t_valid, e_valid = split_time_event(y_valid)
    if noisy:
      experiment_utils.set_seed(seed)
      noisy_x_test = add_gaussian_noise(X_test)
      return df_train, X_train, X_valid, noisy_x_test, t_train, t_valid, t_test, e_train, e_valid, e_test

    return df_train, X_train, X_valid, X_test, t_train, t_valid, t_test, e_train, e_valid, e_test
    
def getPredictions(dataset, params, config, loader=False, sampling=False, noisy=False):
    layers = []
    for _ in range(params['num_layers']):
      layers.append(params['hidden_dim'])
    U = params['U']
    lr = params['lr']
    batch_size = params['batch_size']
    num_epochs = params['n_epochs']
    beta = params['beta']
    alpha = params['alpha']
    samples_tr = params['samples_tr']
    samples_te = params['samples_te']
    seed = params['seed']
    temperature = params['temperature']
    temp_test = params['temp_test']
    patience = params['patience']
    weight_decay = params['weight_decay']

    config['n_samples_train'], config['U'], config['n_samples_test'], config['temperature'], config['temp_test'] = samples_tr, U, samples_te, temperature, temp_test
    config['beta'], config['alpha'], config['seed'] = beta, alpha, seed
    config['dropout_rate'], config['eta'], config['optimizer'] = params['dropout_rate'], params['eta'], params['optimizer']

    experiment_utils.set_seed(seed)

    if loader:
        _, x_train, x_valid, x_test, t_train, t_valid, t_test, e_train, e_valid, e_test = load(dataset, ctype='All', seed=seed, noisy=noisy)
        print(f"x_train: {x_train.shape}, x_valid: {x_valid.shape}, x_test: {x_test.shape}")
    else:
      x_train, x_valid, x_test, t_train, t_valid, t_test, e_train, e_valid, e_test = dataset
    t = np.concatenate((t_train, t_valid, t_test), axis=0).astype('int')
    e = np.concatenate((e_train, e_valid, e_test), axis=0)
    t_max = t.max()
    times = np.linspace(t.min(), t.max(), 100)
    del t, e

    minmax = lambda x: x / t_max # Enforce to be inferior to 1
    eps = 1e-5
    t_train_ = minmax(t_train) #+ eps
    t_valid_ = minmax(t_valid) #+ eps
    # t_test = minmax(t_test) #+ eps
    times_ = minmax(times) #+ eps

    argsortttest = np.argsort(t_test)
    t_test = t_test[argsortttest]
    e_test = e_test[argsortttest]
    x_test = x_test[argsortttest,:]

    from time import time

    model = BayesianSparseNeuralSurvival(patience=patience, layers=layers, config=config, mlayers=1)
    train_start_time = time()
    model.fit(x_train, t_train_, e_train, n_iter = num_epochs, bs = batch_size,
      lr = lr, val_data = (x_valid, t_valid_, e_valid), weight_decay = weight_decay, random_state=seed)
    # nll = model.compute_nll(x_valid, t_valid, e_valid)
    train_time = time() - train_start_time
    print(f"Training time: {train_time}")

    et_train = np.array([(e_train[i] == 1, t_train[i]) for i in range(len(e_train))],
                        dtype = [('e', bool), ('t', float)])
    et_test = np.array([(e_test[i] == 1, t_test[i]) for i in range(len(e_test))],
      dtype = [('e', bool), ('t', float)])

    test_start_time = time()
    if sampling:
      out_survival, out_risk = utils.get_prediction_2(model, x_test.astype(float), times_ddh=times_, samples=config['n_samples_test'], sample=True)
    else:
      out_survival, out_risk = utils.get_prediction_2(model, x_test.astype(float), times_ddh=times_, samples=config['n_samples_test'], sample=False)
    test_time = time() - test_start_time
    print(f"Testing time: {test_time}\n")

    surv_preds = pd.DataFrame(out_survival, columns=times)

    # Sanitize
    surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
    bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
    sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
    sanitized_et_test = np.delete(et_test, bad_idx, axis=0)

    ev = EvalSurv(sanitized_surv_preds.T, sanitized_et_test["t"], sanitized_et_test["e"], censor_surv="km")
    inbll = ev.integrated_nbll(times)
    ci = ev.concordance_td()
    Ibs = ev.integrated_brier_score(times)
    return out_survival, et_train, et_test, times, Ibs

def run_experiments(dataset, seeds, params, config, loader=False, sampling=False, noisy=False):
  metrics = {'CI': [], 'MAE_H': [], 'IBS': [], 'D-Cal':[], 'CI_lif': [], 'IBS_EV': []}
  for seed_exp in seeds:
      params['seed'] = seed_exp
      print(f"********************* Experiment with Seed: {params['seed']} ******************************")

      predictions, et_train_out, et_test_out, time_grid, ibs_ev = getPredictions(dataset=dataset, params=params, config=config, sampling=sampling, loader=loader, noisy=noisy)
      t_test, e_test, t_train, e_train = et_test_out['t'], et_test_out['e'], et_train_out['t'], et_train_out['e']
      # Sanitize
      surv_preds = pd.DataFrame(predictions, columns=time_grid)
      surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
      bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
      sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
      # print(f"Sanitized surv_preds: {sanitized_surv_preds.shape}, surv_preds initial: {surv_preds.shape}")
      sanitized_et_test = np.delete(et_test_out, bad_idx, axis=0)

      lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_et_test["t"], sanitized_et_test["e"], t_train, e_train)
      cindex_lifelines, concordant_pairs, total_pairs = lifelines_eval.concordance(ties="None")
      #print("Concordance index(lifelines_eval) is {}, meaning that the model can correctly order {} pairs among {} comparable pairs "
            #"in the testing set.\n".format(cindex_lifelines, concordant_pairs, total_pairs))

      ibs = lifelines_eval.integrated_brier_score()
      p_value, bin_statistics = lifelines_eval.d_calibration() # 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0

      ev = EvalSurv(sanitized_surv_preds.T, sanitized_et_test["t"], sanitized_et_test["e"], censor_surv="km")
      cindex = ev.concordance_td()

      mae_score_hg = lifelines_eval.mae(method="Hinge")
      #print("MAE-Hinge loss is {}.".format(mae_score_hg))
      
      #metrics['CI'].append(cindex_lifelines)
      #metrics['IBS'].append(ibs_ev)
      metrics['CI'].append(cindex)
      metrics['CI_lif'].append(cindex_lifelines)
      metrics['IBS'].append(ibs)
      metrics['IBS_EV'].append(ibs_ev)
      metrics['MAE_H'].append(mae_score_hg)

      if p_value >= 0.05:
          print(f"\033[32m The model is d-calibrated with C_index = {cindex:.4f} and IBS = {ibs:.4f}!\n\033[0m")
          metrics['D-Cal'].append(1)
      else:
          print(f"The model is not d-calibrated with C_index = {cindex:.4f} and IBS = {ibs:.4f}!\n")
          metrics['D-Cal'].append(0)

  for key, value in metrics.items():
  	if key == 'D-Cal':
  		print(f"{key} : {int(np.sum(value))}/5")
  	else:
    		print(f"{key} = {np.mean(value):.4f}({np.std(value):.4f})")
  return metrics

def main():
	config_te = {'risks': 1, 'rho_scale': -5.0, 'mu_scale': None, 'sigma1': 1, 'sigma2': math.exp(-6), 'pi': 0.5}
	# Reported in the papper
	params = {'lr': 0.0001488788692962846, 'weight_decay': 0.0001, 'optimizer': 'Adam', 'batch_size': 256, 'hidden_dim': 64, 'num_layers': 2, 'n_epochs': 113, 'beta': 0.9866302258496104,
		  'alpha': 0.007879137158754101, 'eta': 0.0, 'samples_tr': 10, 'samples_te': 10, 'U': 2, 'seed': 123, 'temperature': 0.67, 'temp_test': 0.69, 'dropout_rate': 0.4054512199836226, 'patience': 10}
	dataset = 'TCGA'
	seeds  = [2020, 1234, 2023, 2027, 123]
	seer_metrics = run_experiments(dataset, seeds=seeds, params=params, config=config_te, loader=True, sampling=False)


if __name__ == "__main__":
	main()
	

	
