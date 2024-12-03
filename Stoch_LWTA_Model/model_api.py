from dsm.dsm_api import DSMBase
from Stoch_LWTA_Model.model import BayesianSparseNeuralSurvivalTorch
from Stoch_LWTA_Model import Losses
from Stoch_LWTA_Model.utilities import train_model

import torch
import numpy as np
from tqdm import tqdm

class BayesianSparseNeuralSurvival(DSMBase):

  def __init__(self, cuda = torch.cuda.is_available(), samples=1, patience=3, **params):
      self.params = params
      self.fitted = False
      self.cuda = cuda
      self.samples = samples
      self.patience = patience
      if samples <= 1:
          self.loss = Losses.negative_log_likelihood_loss
      else:
          self.loss = Losses.negative_log_likelihood_loss_2

  def _gen_torch_model(self, inputdim, optimizer, risks):
    model = BayesianSparseNeuralSurvivalTorch(inputdim, **self.params).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adamw", weight_decay = 0.0001,  random_state = 42, **args):
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(x_train.size(1), optimizer, risks = maxrisk)
    model, speed = train_model(model, self.loss,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda == 2, patience_max = self.patience, weight_decay = weight_decay,
                         **args)

    self.speed = speed # Number of iterations needed to converge
    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e, adversarial=False):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, self.torch_model.seed)
    _, _, _, x_val, t_val, e_val = processed_data
    if self.cuda == 2:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = self.loss(self.torch_model, x_val, t_val, e_val, samples=self.samples)
    return loss if adversarial else loss.item()
    

  def predict_survival(self, x, t, risk = 1, sample=True, samples=1):
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.tensor([t_] * len(x), dtype=torch.float32, device=x.device)
        log_sr, _, _, _ = self.torch_model(x, t_, sample=sample, n_samples=samples)
        outcomes = 1 - (1 - torch.exp(log_sr)) # outcomes = 1 - (-log_sr)
        scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")
