import numpy as np
import tensorflow as tf
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from utility.survival import make_time_bins, calculate_event_times, calculate_percentiles, compute_deterministic_survival_curve
from utility.training import get_data_loader, scale_data, split_time_event
from tools.sota_builder import make_cox_model, make_coxnet_model, make_coxboost_model
from tools.sota_builder import make_rsf_model, make_dsm_model, make_dcph_model, make_dcm_model
from tools.sota_builder import make_baycox_model, make_baymtlr_model
from tools.bnn_isd_trainer import train_bnn_model
from utility.bnn_isd_models import make_ensemble_cox_prediction, make_ensemble_mtlr_prediction
from pathlib import Path
import paths as pt
import joblib
from time import time
from utility.config import load_config
from pycox.evaluation import EvalSurv
import torch
from utility.survival import survival_probability_calibration
from tools.evaluator import LifelinesEvaluator
import math
from utility.survival import coverage
from scipy.stats import chisquare
from utility.risk import InputFunction
from utility.training import make_stratified_split
from utility.survival import convert_to_structured
from tools.Evaluations.util import make_monotonic, check_monotonicity

# ... (keep other imports and settings)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

def load_and_preprocess(dataset_name):
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()

    X = df[cat_features + num_features]
    y = convert_to_structured(df["time"], df["event"])

    if dataset_name != "TCGA":
        X = scale_data(X_train=X, cat_features=cat_features, num_features=num_features)

    #X = np.array(X).astype(float) # initially not commented
    t, e = split_time_event(y)

    return X, t, e, num_features, cat_features

def run_cross_validation(dataset_name, model_name, n_splits=5, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # Load data
    X, t, e, num_features, cat_features = load_and_preprocess(dataset_name)
    
    # Create a stratification target based on time and event
    le = LabelEncoder()

    # Prepare data for stratification
    t_series = pd.Series(t)
    e_series = pd.Series(e)
    time_bins = pd.qcut(t_series, q=5).astype(str)
    # stratify_target = time_bins + '_' + e_series.astype(str)
    # Combine the time bins and event information using pandas string concatenation
    combined_str = time_bins.astype(str) + '_' + e_series.astype(str)

    # Encode the combined string as integers
    stratify_target = le.fit_transform(combined_str)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize results dataframe
    results = pd.DataFrame()

    for fold, (train_index, test_index) in enumerate(skf.split(X, stratify_target)):
        print(f"Processing fold {fold + 1}/{n_splits}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        t_train, t_test = t[train_index], t[test_index]
        e_train, e_test = e[train_index], e[test_index]
        y_test = convert_to_structured(t_test, e_test)

        # Further split train into train and validation
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed).split(X_train, stratify_target[train_index])
        train_index_inner, val_index = next(train_val_split)

        X_train, X_valid = X_train.iloc[train_index_inner], X_train.iloc[val_index]
        t_train, t_valid = t_train[train_index_inner], t_train[val_index]
        e_train, e_valid = e_train[train_index_inner], e_train[val_index]
        y_train = convert_to_structured(t_train, e_train)
        y_valid = convert_to_structured(t_valid, e_valid)
        
        # Convert to array
		X_train_arr = np.array(X_train)
		X_valid_arr = np.array(X_valid)
		X_test_arr = np.array(X_test)

        # Make time/event split
        t_train, e_train = split_time_event(y_train)
        t_test, e_test = split_time_event(y_test)

        # Make event times
        event_times = calculate_event_times(t_train, e_train)
        mtlr_times = make_time_bins(t_train, event=e_train)

        # Calculate quantiles
        event_times_pct = calculate_percentiles(event_times)
        mtlr_times_pct = calculate_percentiles(mtlr_times)

        # Load config and train model
        if model_name in ["baycox", "baymtlr"]:
            config = dotdict(load_config(getattr(pt, f"{model_name.upper()}_CONFIGS_DIR"), f"{dataset_name.lower()}.yaml"))
        else:
            config = load_config(getattr(pt, f"{model_name.upper()}_CONFIGS_DIR"), f"{dataset_name.lower()}.yaml")

        # Train model (use the existing training logic for each model)
        # ... (keep the existing model training code here)
        print(f"Training {model_name}")
        # Get batch size for MLP to use for loss calculation
        mlp_config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        batch_size = mlp_config['batch_size']
        
        if model_name in ["baycox", "baymtlr"]:
            # Make data for BayCox/BayMTLR models
            data_train = X_train.copy()
            data_train["time"] = pd.Series(y_train['time'])
            data_train["event"] = pd.Series(y_train['event']).astype(int)
            data_valid = X_valid.copy()
            data_valid["time"] = pd.Series(y_valid['time'])
            data_valid["event"] = pd.Series(y_valid['event']).astype(int)
            data_test = X_test.copy()
            data_test["time"] = pd.Series(y_test['time'])
            data_test["event"] = pd.Series(y_test['event']).astype(int)
            num_features = X_train.shape[1]
                    
        # Make model and train
        if model_name == "cox":
            config = load_config(pt.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_cox_model(config)
            train_start_time = time()
            model.fit(X_train_arr, y_train)
            train_time = time() - train_start_time  
        elif model_name == "coxnet":
            config = load_config(pt.COXNET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_coxnet_model(config)
            train_start_time = time()
            model.fit(X_train_arr, y_train)
            train_time = time() - train_start_time
        elif model_name == "dsm":
            config = load_config(pt.DSM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            train_start_time = time()
            model = make_dsm_model(config)
            model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
            train_time = time() - train_start_time
        elif model_name == "dcph":
            config = load_config(pt.DCPH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_dcph_model(config)
            train_start_time = time()
            model.fit(X_train, t_train, e_train, batch_size=config['batch_size'],
                      iters=config['iters'], val_data=(X_valid, t_valid, e_valid),
                      learning_rate=config['learning_rate'], optimizer=config['optimizer'])
            train_time = time() - train_start_time
        elif model_name == "dcm":
            config = load_config(pt.DCM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_dcm_model(config)
            train_start_time = time()
            model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
            train_time = time() - train_start_time
        elif model_name == "rsf":
            config = load_config(pt.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_rsf_model(config)
            train_start_time = time()
            model.fit(X_train_arr, y_train)
            train_time = time() - train_start_time
        elif model_name == "coxboost":
            config = load_config(pt.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            model = make_coxboost_model(config)
            train_start_time = time()
            model.fit(X_train_arr, y_train)
            train_time = time() - train_start_time
        elif model_name == "baycox":
            config = dotdict(load_config(pt.BAYCOX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
            model = make_baycox_model(num_features, config)
            train_start_time = time()
            model = train_bnn_model(model, data_train, data_valid, mtlr_times,
                                    config=config, random_state=0, reset_model=True, device=device)
            train_time = time() - train_start_time
        elif model_name == "baymtlr":
            config = dotdict(load_config(pt.BAYMTLR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
            model = make_baymtlr_model(num_features, mtlr_times, config)
            train_start_time = time()
            model = train_bnn_model(model, data_train, data_valid,
                                    mtlr_times, config=config,
                                    random_state=0, reset_model=True, device=device)
            train_time = time() - train_start_time

        # Compute survival function
        # ... (keep the existing prediction code here)
        test_start_time = time()
        if model_name == "dsm":
            surv_preds = model.predict_survival(X_test, times=list(event_times))
        elif model_name == "dcph":
            surv_preds = model.predict_survival(X_test, t=list(event_times))
        elif model_name == "dcm":
            surv_preds = model.predict_survival(X_test, times=list(event_times))
        elif model_name == "rsf": # uses KM estimator instead
            test_surv_fn = model.predict_survival_function(X_test_arr)
            surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
        elif model_name == "coxboost":
            test_surv_fn = model.predict_survival_function(X_test_arr)
            surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
        elif model_name == "baycox":
            baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                            dtype=torch.float, device=device)
            survival_outputs, _, ensemble_outputs = make_ensemble_cox_prediction(model, baycox_test_data, config)
            surv_preds = survival_outputs.numpy()
        elif model_name == "baymtlr":
            baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                            dtype=torch.float, device=device)
            survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction(model, baycox_test_data, mtlr_times, config)
            surv_preds = survival_outputs.numpy()
        else:
            surv_preds = compute_deterministic_survival_curve(model, X_train_arr, X_test_arr,
                                                              e_train, t_train, event_times, model_name)
        test_time = time() - test_start_time
        
        # Check monotonicity
        if not check_monotonicity(surv_preds):
            surv_preds = make_monotonic(surv_preds, event_times, method='ceil')
            
        # Make dataframe
        if model_name == "baymtlr":
            mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
            surv_preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
        else:
            surv_preds = pd.DataFrame(surv_preds, columns=event_times)

        # Check monotonicity and sanitize
        # ... (keep the existing monotonicity check and sanitization code here)
        if not check_monotonicity(surv_preds):
                surv_preds = make_monotonic(surv_preds, event_times, method='ceil')
                
        # Make dataframe
        if model_name == "baymtlr":
            mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
            surv_preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
        else:
            surv_preds = pd.DataFrame(surv_preds, columns=event_times)
            
        # Sanitize
        surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
        bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
        sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
        sanitized_y_test = np.delete(y_test, bad_idx, axis=0)
        sanitized_x_test = np.delete(X_test_arr, bad_idx, axis=0)
        sanitized_t_test = np.delete(t_test, bad_idx, axis=0)
        sanitized_e_test = np.delete(e_test, bad_idx, axis=0)

        # Compute metrics
        lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test["time"],
                                            sanitized_y_test["event"], t_train, e_train)
        ibs = lifelines_eval.integrated_brier_score()
        mae_hinge = lifelines_eval.mae(method="Hinge")
        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
        d_calib = lifelines_eval.d_calibration()[0]
        km_mse = lifelines_eval.km_calibration()
        ev = EvalSurv(sanitized_surv_preds.T, sanitized_y_test["time"],
                      sanitized_y_test["event"], censor_surv="km")
        inbll = ev.integrated_nbll(event_times)
        ci = ev.concordance_td()

        # Calculate C-cal for BNN models
        if model_name in ['baycox', 'baymtlr']:
            # ... (keep existing C-cal calculation)
            c_calib = pvalue[0]
        else:
            c_calib = 0

        # Compute calibration curves
        deltas = dict()
        if model_name != "baymtlr":  # use event times for non-mtlr model
            for t0 in event_times_pct.values():
                _, _, _, deltas_t0 = survival_probability_calibration(sanitized_surv_preds,
                                                                      sanitized_y_test["time"],
                                                                      sanitized_y_test["event"],
                                                                      t0)
                deltas[t0] = deltas_t0
        else:
            for t0 in mtlr_times_pct.values():
                _, _, _, deltas_t0 = survival_probability_calibration(sanitized_surv_preds,
                                                                      sanitized_y_test["time"],
                                                                      sanitized_y_test["event"],
                                                                      t0)
                deltas[t0] = deltas_t0
        ici = deltas[t0].mean()

        # Save fold results
        fold_results = pd.DataFrame({
            "Fold": [fold + 1],
            "CI": [ci],
            "IBS": [ibs],
            "MAEHinge": [mae_hinge],
            "MAEPseudo": [mae_pseudo],
            "DCalib": [d_calib],
            "KM": [km_mse],
            "INBLL": [inbll],
            "CCalib": [c_calib],
            "ICI": [ici],
            "ModelName": [model_name],
            "DatasetName": [dataset_name]
        })
        results = pd.concat([results, fold_results], ignore_index=True)

    # Calculate and add mean and std of metrics across folds
    mean_results = results.groupby("ModelName").agg({
        "CI": ["mean", "std"],
        "IBS": ["mean", "std"],
        "MAEHinge": ["mean", "std"],
        "MAEPseudo": ["mean", "std"],
        "DCalib": ["mean", "std"],
        "KM": ["mean", "std"],
        "INBLL": ["mean", "std"],
        "CCalib": ["mean", "std"],
        "ICI": ["mean", "std"]
    }).reset_index()
    mean_results.columns = ["ModelName", "CI_mean", "CI_std", "IBS_mean", "IBS_std",
                            "MAEHinge_mean", "MAEHinge_std", "MAEPseudo_mean", "MAEPseudo_std",
                            "DCalib_mean", "DCalib_std", "KM_mean", "KM_std",
                            "INBLL_mean", "INBLL_std", "CCalib_mean", "CCalib_std",
                            "ICI_mean", "ICI_std"]
    mean_results["DatasetName"] = dataset_name

    # Count number of DCalib > 0.05
    dcalib_count = (results["DCalib"] > 0.05).sum()
    mean_results["DCalib_count"] = dcalib_count

    # Save cross-validation results
    # output_path = Path.joinpath(pt.RESULTS_DIR, f"cv_results_{dataset_name.lower()}_{model_name.lower()}_{seed}.csv")
    # pd.concat([results, mean_results], ignore_index=True).to_csv(output_path, index=False)
    output_path = Path.joinpath(pt.RESULTS_DIR, f"cv_results_{dataset_name.lower()}_{seed}.csv")
    mean_results.to_csv(output_path, index=False)

    return results, mean_results

# Usage
DATASETS = ["GBSG2", "WHAS500", "METABRIC", "SEER"]
MODELS = ["cox", "coxnet", "coxboost", "rsf", "dsm", "dcm", "baycox", "baymtlr"]

for dataset_name in DATASETS:
    for model_name in MODELS:
        print(f"Running cross-validation for {model_name} on {dataset_name}")
        _,_ = run_cross_validation(dataset_name, model_name, n_splits=5, seed=0)