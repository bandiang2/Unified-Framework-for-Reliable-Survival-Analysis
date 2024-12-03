import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)	#[333,333]
	indicator_matrix = torch.tril(matrix_ones)

	return(indicator_matrix)

def rank_loss(pred, ytime, yevent):
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)

	if torch.cuda.is_available():
		ytime_indicator =ytime_indicator.cuda()

	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	if torch.cuda.is_available():
		zeros = torch.zeros(n_sample).cuda()
	else: zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	
	pred_matrix=(pred.reshape(1,-1)-pred).mul(ytime_indicator)
	pred_matrix=nn.ReLU()(torch.sign(pred_matrix))

	pred_matrix = pred_matrix.mul(ytime_matrix)

	rankloss=torch.sum(pred_matrix)
	return rankloss
	
def negative_log_likelihood_loss(model, x, t, e, samples=1):
    # Go through network
    log_sr, tau, kl_total, ib = model.forward(x, t)
    log_hr = model.gradient(log_sr, tau, e).log()
    # print(f'log_sr: {log_sr.shape}')

    error = 0
    for k in range(model.risks):
        error -= log_sr[e != (k + 1)][:, k].sum()
        error -= log_hr[e == (k + 1)].sum()
    
    # pred = 1 - (1 - torch.exp(log_sr))
    r_loss = 0
    # r_loss += rank_loss(1 - (-log_sr), t, e)
    return error / len(x)  + (model.eta * r_loss) + (model.alpha * kl_total) + (model.beta * ib)  #(error + (model.alpha * kl_total) + (model.beta * ib)) / len(x)


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

# def negative_log_likelihood_loss(model, x, t, e, samples=1):
#   # Go through network
#   log_sr, tau, kl_total, ib = model.forward(x, t)
#   log_hr = model.gradient(log_sr, tau, e).log()

#   # Likelihood error
#   error = 0
#   for k in range(model.risks):
#     error -= log_sr[e != (k + 1)][:, k].sum()
#     error -= log_hr[e == (k + 1)].sum()

#   return error / len(x) + (model.alpha * kl_total) + (model.beta * ib) # (error + (model.alpha * kl_total) + (model.beta * ib)) / len(x)


# def negative_log_likelihood_loss_2(model, x, t, e, samples=5):
#   # Go through network
#   log_sr, tau, kl_div, info_bound = model.sample_elbo(x, t, samples=samples)
#   log_hr = model.gradient(log_sr, tau, e).log()

#   # Likelihood error
#   error = 0
#   for k in range(model.risks):
#     error -= log_sr[e != (k + 1)][:, k].sum()
#     error -= log_hr[e == (k + 1)].sum()

#   return (error + (model.alpha * kl_div) + (model.beta * info_bound)) / len(x)
