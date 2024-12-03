import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def negative_log_likelihood_loss(model, x, t, e, samples=1):
  # Go through network
  log_sr, tau, kl_total, ib = model.forward(x, t)
  log_hr = model.gradient(log_sr, tau, e).log()

  # Likelihood error
  error = 0
  for k in range(model.risks):
    error -= log_sr[e != (k + 1)][:, k].sum()
    error -= log_hr[e == (k + 1)].sum()

  return error / len(x) + (model.alpha * kl_total) + (model.beta * ib) # (error + (model.alpha * kl_total) + (model.beta * ib)) / len(x)


def negative_log_likelihood_loss_2(model, x, t, e, samples=5):
  # Go through network
  log_sr, tau, kl_div, info_bound = model.sample_elbo(x, t, samples=samples)
  log_hr = model.gradient(log_sr, tau, e).log()

  # Likelihood error
  error = 0
  for k in range(model.risks):
    error -= log_sr[e != (k + 1)][:, k].sum()
    error -= log_hr[e == (k + 1)].sum()

  return (error + (model.alpha * kl_div) + (model.beta * info_bound)) / len(x)
