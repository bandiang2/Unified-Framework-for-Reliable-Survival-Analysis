import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

def convert_to_structured(T, E):
    # dtypes for conversion
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}

    # concat of events and times
    concat = list(zip(E, T))

    # return structured array
    return np.array(concat, dtype=default_dtypes)
    
def drop_constants(data):
    mask = (data != data.iloc[0]).any()
    return data.loc[:, mask]


def filter_modalities(data, selected_modalities_ix, all_modalities):
    modalities_to_keep_ix = np.array(
        [int(i) for i in selected_modalities_ix.rsplit(",")]
    )
    all_modalities = np.array(all_modalities)
    modalities_to_keep = all_modalities[modalities_to_keep_ix]
    modality_mask = [
        col for col in data.columns if col.rsplit("_")[0] in modalities_to_keep
    ]
    return data[modality_mask]

def get_transform(data, test_size, seed=42, EPS=1e-8):
    time, event = data["OS_days"].astype(int), data["OS"].astype(int)
    event = event + EPS
    data = data.drop(columns=["OS_days", "OS"])
    y = convert_to_structured(time, event)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, shuffle=True, stratify=y['event'], random_state=seed)
    ct = ColumnTransformer(
                    [
                        (
                            "numerical",
                            make_pipeline(StandardScaler()),
                            np.where(X_train.dtypes != "object")[0],
                        ),
                        (
                            "categorical",
                            make_pipeline(
                                OneHotEncoder(
                                    sparse=False, handle_unknown="ignore"
                                ),
                                StandardScaler(),
                            ),
                            np.where(X_train.dtypes == "object")[0],
                        ),
                    ]
                )
    X_train = ct.fit_transform(X_train)
    X_train = pd.DataFrame(
                        X_train,
                        columns=data.columns[
                            np.where(data.dtypes != "object")[0]
                        ].tolist()
                        + [
                            f"clinical_{i}"
                            for i in ct.transformers_[1][1][0]
                            .get_feature_names_out()
                            .tolist()
                        ],
                    )
    X_test = pd.DataFrame(
                ct.transform(X_test), columns=X_train.columns
            )
    
    return X_train, y_train, X_test, y_test
    

modalities = "0,1,3,6"
modality_order = ["clinical", "gex", "rppa", "mirna", "mutation", "meth", "cnv"]

def load(dataset, test_size=0.2, seed=42):
    file = './sample_data/'+dataset.upper()+'_data_preprocessed.csv'
    # data = pd.read_csv(file).dropna()
    data = pd.read_csv(file).fillna("MISSING")
    return get_transform(data, test_size, seed)
    