import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500, load_flchain
from sklearn.model_selection import train_test_split
#import shap
from abc import ABC, abstractmethod
from typing import Tuple, List
from tools.preprocessor import Preprocessor
import paths as pt
from pathlib import Path
from utility.survival import convert_to_structured

# added
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.utils import check_random_state
from sklearn.datasets import make_friedman1

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None

    @abstractmethod
    def load_data(self) -> None:
        """Loads the data from a data set at startup"""

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train['Time'])
        t_valid = np.array(y_valid['Time'])
        t_test = np.array(y_test['Time'])
        e_train = np.array(y_train['Event'])
        e_valid = np.array(y_valid['Event'])
        e_test = np.array(y_test['Event'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :return: df
        """
        df = pd.DataFrame(self.X)
        df['time'] = self.y['time']
        df['event'] = self.y['event']
        return df

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['category']).columns.tolist()

    def prepare_data(self, train_size: float = 0.7) -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """
        This method prepares and splits the data from a data set
        :param train_size: the size of the train set
        :return: a split train and test dataset
        """
        X = self.X
        y = self.y
        cat_features = self.cat_features
        num_features = self.num_features

        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=0)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=0)

        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
        transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        X_train = transformer.transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

        X_train = np.array(X_train, dtype=np.float32)
        X_valid = np.array(X_valid, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

## Added new
class TcgaDataLoader(BaseDataLoader):
    """
    Data loader for TCGA dataset
    """

    def __init__(self, type='BRCA'):
        super(TcgaDataLoader, self).__init__()
        self.type = type
    

    def load_data(self):
        # skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        # cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        # data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
        
        # Cancer type embedded index
        cancer_type_dic = {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'COAD': 5, 'DLBC': 6, 'ESCA': 7, 'GBM': 8, 'HNSC': 9,
                        'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LAML': 13, 'LGG': 14, 'LIHC': 15, 'LUAD': 16, 'LUSC': 17, 'MESO': 18,
                        'OV': 19, 'PAAD': 20, 'PCPG': 21, 'PRAD': 22, 'READ': 23, 'SARC': 24, 'SKCM': 25, 'STAD': 26, 'TGCT': 27,
                        'THCA': 28, 'THYM': 29, 'UCEC': 30, 'UCS': 31, 'UVM': 32}

        cnv_path = Path.joinpath(pt.DATA_DIR, "PC_CNV_threshold_20.csv")
        mirna_path = Path.joinpath(pt.DATA_DIR, "PC_miRNA.csv")
        mrna_path = Path.joinpath(pt.DATA_DIR, "PC_mRNA_threshold_7.csv")
        cli_path = Path.joinpath(pt.DATA_DIR, "Pc_clinical_emb.csv")
        
        cnv_data = pd.read_csv(cnv_path, header=None)
        dummy_names = [f'X_{i}_cnv' for i in range(len(cnv_data.columns))]
        cnv_data.columns = dummy_names
        # print(f"CNV : {cnv_data.shape}\n")
        
        # mirna_data = pd.read_csv(mirna_path, header=None)
        # dummy_names = [f'X_{i}_mirna' for i in range(len(mirna_data.columns))]
        # mirna_data.columns = dummy_names
        # print(f"MiRNA : {mirna_data.shape}\n")
        
        mrna_data = pd.read_csv(mrna_path, header=None)
        dummy_names = [f'X_{i}_mrna' for i in range(len(mrna_data.columns))]
        mrna_data.columns = dummy_names
        # print(f"MRNA : {mrna_data.shape}\n")
        
        columns = ['id', 'cancer_type', 'gender', 'race',
                   'histological_type', 'age', 'event', 'time']
        clin_data = pd.read_csv(cli_path, names=columns)
        # print(f"Clinical : {clin_data.shape}\n")
        
        data = pd.concat([clin_data, cnv_data, mrna_data], axis=1).dropna()
        data['age'] = data['age'] / data['age'].max()
        data = data[data['time'] > 0].reset_index(drop=True)
        

        if self.type != 'All':
            cancer_type = cancer_type_dic[self.type]
            data = data[data['cancer_type'] == cancer_type]#.reset_index(drop=True)
        columns_drop = ['id', 'race', 'histological_type', 'cancer_type', 'event', 'time']

        outcomes = data.copy()
        outcomes['event'] = data['event']
        outcomes['time'] = data['time']
        outcomes = outcomes[['event', 'time']]

        data = data.drop(columns=columns_drop)
        #print(f"Concat (clinical, cnv, mrna): {data.shape}")
        # print(f"{self.type} (clinical, cnv, mrna) after drop: {data.shape}\n")

        self.X = pd.DataFrame(data)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = []
        # print(f"# of numerical features: {len(self.num_features)}\n")
        
        # print(f"self.X: {self.X.shape}")
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self
        
## Added new
class TcgaDataLoader2(BaseDataLoader):
    """
    Data loader for TCGA dataset
    """

    def __init__(self):
        super(TcgaDataLoader2, self).__init__()
    

    def load_data(self, cancer_type_file):
        file_path = Path.joinpath(pt.DATA_DIR, cancer_type_file)
        data = pd.read_csv(file_path,  encoding='ISO-8859-1', low_memory=False).dropna()
        data = data = data[data['survival_months'] > 0]
        
        outcomes = data.copy()
        outcomes['event'] = data['censorship']
        outcomes['time'] = data['survival_months']
        outcomes = outcomes[['event', 'time']]
        
        columns_drop = ['case_id','slide_id','site','is_female','oncotree_code','train', 'survival_months', 'censorship']
        data = data.drop(columns=columns_drop)
        
        print(f"data shape: {data.shape}")

        self.X = pd.DataFrame(data)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = []
        # print(f"# of numerical features: {len(self.num_features)}\n")
        
        # print(f"self.X: {self.X.shape}")
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

def make_mimic_data() -> pd.DataFrame:
    path = Path.joinpath(pt.DATA_DIR, 'mimic.csv')
    data = pd.read_csv(path)
    # data = pd.read_csv('data/MIMIC/MIMIC_IV_v2.0_preprocessed.csv')
    skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    return data

def make_nacd_data() -> pd.DataFrame:
    path = Path.joinpath(pt.DATA_DIR, 'NACD_Full.csv')
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = pd.read_csv(path).drop(cols_to_drop, axis=1).rename(columns={"#NAME?": "event", "SURVIVAL":"time"})

    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    return data

class NacdDataLoader(BaseDataLoader):
    """
    Data loader for NACD dataset
    """
    def load_data(self):
        data = make_nacd_data()
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['time']
        outcomes = outcomes[['event', 'time']]

        data = data.drop(['event', "time"], axis=1)

        self.X = pd.DataFrame(data)

        self.num_features = self.X.columns.to_list()
        self.cat_features = []
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

# class MimicDataLoader(BaseDataLoader):
#     """
#     Data loader for MIMIC dataset
#     """
#     def load_data(self):
#         data = make_mimic_data()
        
#         outcomes = data.copy()
#         outcomes['event'] =  data['event']
#         outcomes['time'] = data['time']
#         outcomes = outcomes[['event', 'time']]

#         data = data.drop(['event', "time"], axis=1)

#         self.X = pd.DataFrame(data)

#         self.num_features = self.X.columns.to_list()
#         self.cat_features = []
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

class MimicDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self):
        #skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        #cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        #data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
        
        path = Path.joinpath(pt.DATA_DIR, "mimic.csv")
        data = pd.read_csv(path)
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['time']
        outcomes = outcomes[['event', 'time']]

        data = data.drop(['event', "time"], axis=1)
        
        obj_cols = ['is_male', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        # data[obj_cols] = data[obj_cols].astype('object')
        # data[obj_cols] = data[obj_cols].astype('category')

        self.X = pd.DataFrame(data)

        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

class SeerDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset
    """
    def load_data(self):
        path = Path.joinpath(pt.DATA_DIR, 'seer.csv')
        data = pd.read_csv(path)

        data = data.loc[data['Survival Months'] > 0]
        
        numeric_rows = pd.to_numeric(data["Grade"], errors='coerce').notna()
        data = data[numeric_rows]

        outcomes = data.copy()
        outcomes['event'] =  data['Status']
        outcomes['time'] = data['Survival Months']
        outcomes = outcomes[['event', 'time']]
        outcomes.loc[outcomes['event'] == 'Alive', ['event']] = 0
        outcomes.loc[outcomes['event'] == 'Dead', ['event']] = 1

        data = data.drop(['Status', "Survival Months"], axis=1)

        obj_cols = data.select_dtypes(['bool']).columns.tolist() \
                + data.select_dtypes(['object']).columns.tolist()
        # print(f"obj_cols: {obj_cols}") this works
        for col in obj_cols:
            data[col] = data[col].astype('object')#

        self.X = pd.DataFrame(data)

        self.num_features = self._get_num_features(self.X)
        self.cat_features = obj_cols #self._get_cat_features(self.X)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

# class SupportDataLoader(BaseDataLoader):
#     """
#     Data loader for SUPPORT dataset
#     """
#     def load_data(self):
#         path = Path.joinpath(pt.DATA_DIR, 'support.feather')
#         data = pd.read_feather(path)
#         # path = Path.joinpath(pt.DATA_DIR, 'support2.csv')
#         # data = pd.read_csv(path)

#         data = data.loc[data['duration'] > 0]
#         # data = data.loc[data['d.time'] > 0]

#         outcomes = data.copy()
#         outcomes['event'] =  data['event']
#         outcomes['time'] = data['duration']
#         outcomes = outcomes[['event', 'time']]
#         # outcomes['event'] =  data['death']
#         # outcomes['time'] = data['d.time']
#         # outcomes = outcomes[['event', 'time']]

#         num_feats =  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
#                       'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
#         # num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
#         #      'urine', 'adlp', 'adls']
#         # obj_cols = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
#         # for col in obj_cols:
#         #     data[col] = data[col].astype('object')
        

#         self.num_features = num_feats
#         self.cat_features = []
#         # self.cat_features = obj_cols
#         self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
#         # self.X = pd.DataFrame(data)
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

class SupportDataLoader(BaseDataLoader):
    """
    Data loader for SUPPORT dataset
    """
    def load_data(self):
        data = make_support_data()
        
        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.num_features = self.X.columns.to_list()
        self.cat_features = []
        self.y = convert_to_structured(data['time'].values.astype(float), data['event'].values.astype(int))

        return self


'''
class NhanesDataLoader(BaseDataLoader):
    """
    Data loader for NHANES dataset
    """
    def load_data(self):
        X, y = shap.datasets.nhanesi()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        event = np.array([True if x > 0 else False for x in y])
        time = np.array(abs(y))
        self.y = convert_to_structured(time, event)

        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self
'''

class AidsDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_aids()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
        self.X = pd.DataFrame(X)

        self.y = convert_to_structured(y['time'], y['censor'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class GbsgDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        X, y = load_gbsg2()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['time'], y['cens'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class WhasDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_whas500()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['lenfol'], y['fstat'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class FlchainDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_flchain()
        X['event'] = y['death']
        X['time'] = y['futime']

        X = X.loc[X['time'] > 0]
        self.y = convert_to_structured(X['time'], X['event'])
        X = X.drop(['event', 'time'], axis=1).reset_index(drop=True)

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('object')

        self.X = pd.DataFrame(X)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

class MetabricDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        path = Path.joinpath(pt.DATA_DIR, 'metabric.feather')
        data = pd.read_feather(path) 
        data['duration'] = data['duration'].apply(round)

        data = data.loc[data['duration'] > 0]
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x8'] \
                     + ['x4', 'x5', 'x6', 'x7']

        self.num_features = num_feats
        self.cat_features = []
                    
        self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

class SyntheticDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        df, coef = generate_synthetic_data(type='poly')
        # drop_cols = ["time", "event", "true_time"]
        X = df.drop(["time", "event", "true_time"], axis=1)

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(df['time'].values.astype(float), df['event'].values.astype(int))
        self.num_features = X.columns.to_list()
        self.cat_features = []
        return self

def make_synthetic_data(
        n_samples: int = 10000,
        n_noise_features: int = 47,
        base_hazard: int = 0.1,
        percent_censor: float = 0.3
) -> (pd.DataFrame, np.ndarray):
    """Generates a synthetic survival dataset with linear hazard. (Borrowed form torchmtlr)"""
    x = np.random.standard_normal((n_samples, n_noise_features + 3))
    hazards = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 2]
    event_time = np.random.exponential(1 / (base_hazard * np.exp(hazards)))
    censor_time = np.quantile(event_time, 1 - percent_censor)

    time = np.minimum(event_time, censor_time)
    event = (event_time < censor_time).astype(np.int)

    df = pd.DataFrame({
        "time": time,
        "event": event,
        "true_time": event_time,
        **{f"x{i+1}": x[:, i] for i in range(x.shape[1])}
    })
    return df, np.array([1, 2, -0.5])


def generate_synthetic_data(
        censor_dist: str = 'Uniform',
        n_samples: int = 10000,
        n_features: int = 10,
        type: str = 'linear'
) -> (pd.DataFrame, np.ndarray):
    if type == "linear":
        X, true_times, coef = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5,
                                              bias=0, noise=0.05, random_state=None)
    elif type == "poly":
        X, true_times = make_friedman1(n_samples, n_features=n_features, noise=0.05)
        coef = None
    else:
        raise NotImplementedError

    true_times = true_times.round(decimals=1)
    X = X.round(decimals=1)
    # make sure the dataset is positive
    if true_times.min() < 0:
        true_times += -true_times.min() + 0.1
    times = np.copy(true_times)

    if censor_dist == "Uniform":
        event_status = np.ones(n_samples)
        censor_time = np.random.uniform(low=true_times.min(), high=true_times.max(), size=n_samples).round(decimals=1)

        event_status[censor_time < true_times] = 0
        times[event_status == 0] = censor_time[event_status == 0]
        df = pd.DataFrame({
            "time": times,
            "event": event_status,
            "true_time": true_times,
            **{f"x{i+1}": X[:, i] for i in range(X.shape[1])}
        })
        return df, coef
    else:
        raise NotImplementedError


def make_regression(
        n_samples=10000,
        n_features=100,
        n_informative=20,
        bias=0.0,
        noise=0.05,
        shuffle=True,
        random_state=None
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Generate a random regression problem. (Borrowed from sklearn)

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.

    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=100
        The number of features.

    n_informative : int, default=10
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    bias : float, default=0.0
        The bias term in the underlying linear model.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.

    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
    """
    generator = check_random_state(random_state)

    # Randomly generate a well conditioned input set
    X = generator.randn(n_samples, n_features)

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros((n_features, 1))
    ground_truth[:n_informative, :] = 2.5 * generator.rand(n_informative, 1)

    y = np.dot(X, ground_truth) + bias

    # Add noise
    assert noise > 0.0, "The standard deviation of the noise must higher than 0"
    y += generator.normal(loc=0.0, scale=noise, size=y.shape)

    # Randomly permute samples and features
    if shuffle:
        X, y = sklearn_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)
    return X, y, np.squeeze(ground_truth)

def make_support_data():
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://biostat.app.vumc.org/wiki/Main/SupportDesc for more information.

    Returns
    -------
    pd.DataFrame
        DataFrame with processed covariates for one patient in each row.

    References
    ----------
    ..[1] W. A. Knaus et al., ‘The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults’, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = [
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
        "sfdm2",
        "adlp",     # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
        # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
        "adls",
        "dzgroup",
        "sps",
        "aps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "hday",
    ]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    data = (pd.read_csv(url)
            .drop(cols_to_drop, axis=1)
            .rename(columns={"d.time": "time", "death": "event"}))
    data["event"] = data["event"].astype(int)

    data["ca"] = (data["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        "edu": data["edu"].mean(),
        "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(),
        "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(),
        "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(),
        "temp": data["temp"].mean(),
        "sod": data["sod"].mean(),
        "income": data["income"].mode()[0],
        "race": data["race"].mode()[0],
    }
    data = data.fillna(fill_vals)

    data.sex.replace({'male': 1, 'female': 0}, inplace=True)
    data.income.replace({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}, inplace=True)
    skip_cols = ['event', 'sex', 'time', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    return data



#*************************************************************************************************************************************************************************************
# import numpy as np
# import pandas as pd
# from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500, load_flchain
# from sklearn.model_selection import train_test_split
# #import shap
# from abc import ABC, abstractmethod
# from typing import Tuple, List
# from tools.preprocessor import Preprocessor
# import paths as pt
# from pathlib import Path
# from utility.survival import convert_to_structured

# class BaseDataLoader(ABC):
#     """
#     Base class for data loaders.
#     """
#     def __init__(self):
#         """Initilizer method that takes a file path, file name,
#         settings and optionally a converter"""
#         self.X: pd.DataFrame = None
#         self.y: np.ndarray = None
#         self.num_features: List[str] = None
#         self.cat_features: List[str] = None

#     @abstractmethod
#     def load_data(self) -> None:
#         """Loads the data from a data set at startup"""

#     def make_time_event_split(self, y_train, y_valid, y_test) -> None:
#         t_train = np.array(y_train['Time'])
#         t_valid = np.array(y_valid['Time'])
#         t_test = np.array(y_test['Time'])
#         e_train = np.array(y_train['Event'])
#         e_valid = np.array(y_valid['Event'])
#         e_test = np.array(y_test['Event'])
#         return t_train, t_valid, t_test, e_train, e_valid, e_test

#     def get_data(self) -> pd.DataFrame:
#         """
#         This method returns the features and targets
#         :return: df
#         """
#         df = pd.DataFrame(self.X)
#         df['time'] = self.y['time']
#         df['event'] = self.y['event']
#         return df

#     def get_features(self) -> List[str]:
#         """
#         This method returns the names of numerical and categorial features
#         :return: the columns of X as a list
#         """
#         return self.num_features, self.cat_features

#     def _get_num_features(self, data) -> List[str]:
#         return data.select_dtypes(include=np.number).columns.tolist()

#     def _get_cat_features(self, data) -> List[str]:
#         return data.select_dtypes(['category']).columns.tolist()

#     def prepare_data(self, train_size: float = 0.7) -> Tuple[np.ndarray, np.ndarray,
#                                                              np.ndarray, np.ndarray]:
#         """
#         This method prepares and splits the data from a data set
#         :param train_size: the size of the train set
#         :return: a split train and test dataset
#         """
#         X = self.X
#         y = self.y
#         cat_features = self.cat_features
#         num_features = self.num_features

#         X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=0)
#         X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=0)

#         preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
#         transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
#                                       one_hot=True, fill_value=-1)
#         X_train = transformer.transform(X_train)
#         X_valid = transformer.transform(X_valid)
#         X_test = transformer.transform(X_test)

#         X_train = np.array(X_train, dtype=np.float32)
#         X_valid = np.array(X_valid, dtype=np.float32)
#         X_test = np.array(X_test, dtype=np.float32)

#         return X_train, X_valid, X_test, y_train, y_valid, y_test

# ## Added new
# class TcgaDataLoader(BaseDataLoader):
#     """
#     Data loader for TCGA dataset
#     """

#     def __init__(self, type='BRCA'):
#         super(TcgaDataLoader, self).__init__()
#         self.type = type
    

#     def load_data(self):
#         # skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
#         # cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
#         # data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
        
#         # Cancer type embedded index
#         cancer_type_dic = {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'COAD': 5, 'DLBC': 6, 'ESCA': 7, 'GBM': 8, 'HNSC': 9,
#                         'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LAML': 13, 'LGG': 14, 'LIHC': 15, 'LUAD': 16, 'LUSC': 17, 'MESO': 18,
#                         'OV': 19, 'PAAD': 20, 'PCPG': 21, 'PRAD': 22, 'READ': 23, 'SARC': 24, 'SKCM': 25, 'STAD': 26, 'TGCT': 27,
#                         'THCA': 28, 'THYM': 29, 'UCEC': 30, 'UCS': 31, 'UVM': 32}

#         cnv_path = Path.joinpath(pt.DATA_DIR, "PC_CNV_threshold_20.csv")
#         mirna_path = Path.joinpath(pt.DATA_DIR, "PC_miRNA.csv")
#         mrna_path = Path.joinpath(pt.DATA_DIR, "PC_mRNA_threshold_7.csv")
#         cli_path = Path.joinpath(pt.DATA_DIR, "Pc_clinical_emb.csv")
        
#         cnv_data = pd.read_csv(cnv_path, header=None)
#         dummy_names = [f'X_{i}_cnv' for i in range(len(cnv_data.columns))]
#         cnv_data.columns = dummy_names
#         # print(f"CNV : {cnv_data.shape}\n")
        
#         mirna_data = pd.read_csv(mirna_path, header=None)
#         dummy_names = [f'X_{i}_mirna' for i in range(len(mirna_data.columns))]
#         mirna_data.columns = dummy_names
#         # print(f"MiRNA : {mirna_data.shape}\n")
        
#         mrna_data = pd.read_csv(mrna_path, header=None)
#         dummy_names = [f'X_{i}_mrna' for i in range(len(mrna_data.columns))]
#         mrna_data.columns = dummy_names
#         # print(f"MRNA : {mrna_data.shape}\n")
        
#         columns = ['id', 'cancer_type', 'gender', 'race',
#                   'histological_type', 'age', 'event', 'time']
#         clin_data = pd.read_csv(cli_path, names=columns)
#         # print(f"Clinical : {clin_data.shape}\n")
        
#         # data = pd.concat([clin_data, cnv_data, mrna_data], axis=1).dropna()
#         data = pd.concat([clin_data, mrna_data], axis=1).dropna()
#         data = data[data['time'] > 0].reset_index(drop=True)
#         # print(f"Concat (clinical, cnv, mrna): {data.shape}")

#         cancer_type = cancer_type_dic[self.type]
#         data = data[data['cancer_type'] == cancer_type]#.reset_index(drop=True)
#         columns_drop = ['id', 'cancer_type', 'race',
#                         'histological_type', 'event', 'time']

#         outcomes = data.copy()
#         outcomes['event'] = data['event']
#         outcomes['time'] = data['time']
#         outcomes = outcomes[['event', 'time']]

#         data = data.drop(columns=columns_drop)
#         # print(f"{self.type} (clinical, cnv, mrna) after drop: {data.shape}\n")

#         self.X = pd.DataFrame(data)
#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = []
#         # print(f"# of numerical features: {len(self.num_features)}\n")
        
#         # print(f"self.X: {self.X.shape}")
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

# class MimicDataLoader(BaseDataLoader):
#     """
#     Data loader for MIMIC dataset
#     """
#     def load_data(self):
#         #skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
#         #cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
#         #data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
        
#         path = Path.joinpath(pt.DATA_DIR, "mimic.csv")
#         data = pd.read_csv(path)
        
#         outcomes = data.copy()
#         outcomes['event'] =  data['event']
#         outcomes['time'] = data['time']
#         outcomes = outcomes[['event', 'time']]

#         data = data.drop(['event', "time"], axis=1)
        
#         obj_cols = ['is_male', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
#         # data[obj_cols] = data[obj_cols].astype('category')
#         # data[obj_cols] = data[obj_cols].astype('object')

#         self.X = pd.DataFrame(data)

#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

# class SeerDataLoader(BaseDataLoader):
#     """
#     Data loader for SEER dataset
#     """
#     def load_data(self):
#         path = Path.joinpath(pt.DATA_DIR, 'seer.csv')
#         data = pd.read_csv(path)

#         data = data.loc[data['Survival Months'] > 0]
        
#         numeric_rows = pd.to_numeric(data["Grade"], errors='coerce').notna()
#         data = data[numeric_rows]

#         outcomes = data.copy()
#         outcomes['event'] =  data['Status']
#         outcomes['time'] = data['Survival Months']
#         outcomes = outcomes[['event', 'time']]
#         outcomes.loc[outcomes['event'] == 'Alive', ['event']] = 0
#         outcomes.loc[outcomes['event'] == 'Dead', ['event']] = 1

#         data = data.drop(['Status', "Survival Months"], axis=1)

#         obj_cols = data.select_dtypes(['bool']).columns.tolist() \
#                 + data.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             # data[col] = data[col].astype('category')
#             data[col] = data[col].astype('object')

#         self.X = pd.DataFrame(data)

#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

# class SupportDataLoader(BaseDataLoader):
#     """
#     Data loader for SUPPORT dataset
#     """
#     def load_data(self):
#         path = Path.joinpath(pt.DATA_DIR, 'support.feather')
#         data = pd.read_feather(path)

#         data = data.loc[data['duration'] > 0]

#         outcomes = data.copy()
#         outcomes['event'] =  data['event']
#         outcomes['time'] = data['duration']
#         outcomes = outcomes[['event', 'time']]

#         num_feats =  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
#                       'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

#         self.num_features = num_feats
#         self.cat_features = []
#         self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self

# '''
# class NhanesDataLoader(BaseDataLoader):
#     """
#     Data loader for NHANES dataset
#     """
#     def load_data(self):
#         X, y = shap.datasets.nhanesi()

#         obj_cols = X.select_dtypes(['bool']).columns.tolist() \
#                   + X.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             X[col] = X[col].astype('category')

#         self.X = pd.DataFrame(X)
#         event = np.array([True if x > 0 else False for x in y])
#         time = np.array(abs(y))
#         self.y = convert_to_structured(time, event)

#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         return self
# '''

# class AidsDataLoader(BaseDataLoader):
#     def load_data(self) -> None:
#         X, y = load_aids()

#         obj_cols = X.select_dtypes(['bool']).columns.tolist() \
#                   + X.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             X[col] = X[col].astype('category')
#         self.X = pd.DataFrame(X)

#         self.y = convert_to_structured(y['time'], y['censor'])
#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         return self

# class GbsgDataLoader(BaseDataLoader):
#     def load_data(self) -> BaseDataLoader:
#         X, y = load_gbsg2()

#         obj_cols = X.select_dtypes(['bool']).columns.tolist() \
#                   + X.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             X[col] = X[col].astype('category')

#         self.X = pd.DataFrame(X)
#         self.y = convert_to_structured(y['time'], y['cens'])
#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         return self

# class WhasDataLoader(BaseDataLoader):
#     def load_data(self) -> None:
#         X, y = load_whas500()

#         obj_cols = X.select_dtypes(['bool']).columns.tolist() \
#                   + X.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             X[col] = X[col].astype('category')

#         self.X = pd.DataFrame(X)
#         self.y = convert_to_structured(y['lenfol'], y['fstat'])
#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         return self

# class FlchainDataLoader(BaseDataLoader):
#     def load_data(self) -> None:
#         X, y = load_flchain()
#         X['event'] = y['death']
#         X['time'] = y['futime']

#         X = X.loc[X['time'] > 0]
#         self.y = convert_to_structured(X['time'], X['event'])
#         X = X.drop(['event', 'time'], axis=1).reset_index(drop=True)

#         obj_cols = X.select_dtypes(['bool']).columns.tolist() \
#                   + X.select_dtypes(['object']).columns.tolist()
#         for col in obj_cols:
#             X[col] = X[col].astype('object')

#         self.X = pd.DataFrame(X)
#         self.num_features = self._get_num_features(self.X)
#         self.cat_features = self._get_cat_features(self.X)
#         return self

# class MetabricDataLoader(BaseDataLoader):
#     def load_data(self) -> None:
#         path = Path.joinpath(pt.DATA_DIR, 'metabric.feather')
#         data = pd.read_feather(path) 
#         data['duration'] = data['duration'].apply(round)

#         data = data.loc[data['duration'] > 0]
        
#         outcomes = data.copy()
#         outcomes['event'] =  data['event']
#         outcomes['time'] = data['duration']
#         outcomes = outcomes[['event', 'time']]

#         num_feats =  ['x0', 'x1', 'x2', 'x3', 'x8'] \
#                      + ['x4', 'x5', 'x6', 'x7']

#         self.num_features = num_feats
#         self.cat_features = []
                    
#         self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
#         self.y = convert_to_structured(outcomes['time'], outcomes['event'])

#         return self
