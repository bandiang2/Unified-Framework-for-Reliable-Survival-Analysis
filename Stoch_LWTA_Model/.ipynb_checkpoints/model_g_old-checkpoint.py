import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions import Bernoulli, Exponential, kl_divergence
# from lwta_activations import LWTA

class BayesianLinear(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.0):
        super(BayesianLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # Parameters for mean
        self.weight_mu = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_mu = nn.Parameter(torch.Tensor(output_features))
        
        # Parameters for variance (rho)
        self.weight_rho = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_rho = nn.Parameter(torch.Tensor(output_features))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Prior distribution for all weights and biases
        self.weight_prior = torch.distributions.Normal(0, prior_var)
        self.bias_prior = torch.distributions.Normal(0, prior_var)
        
        # Standard deviation of the prior
        self.prior_var = prior_var

    def reset_parameters(self):
        # Initialize the parameters similar to a standard linear layer
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize rho parameters to provide a reasonable starting point
        nn.init.constant_(self.weight_rho, -5)
        nn.init.constant_(self.bias_rho, -5)

    def forward(self, input, k=1):
        self.input = input
        # Calculate the standard deviation from rho
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho)) #'weight_sigma'
        # print(f"weight_sigma: {weight_sigma}")
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # Sample weights and biases from the posterior
        # if setl.training:
        weight = self.weight_mu + F.softplus(self.weight_sigma, beta=10) * torch.randn_like(self.weight_sigma)
        bias = self.bias_mu + F.softplus(self.bias_sigma, beta=10) * torch.randn_like(self.bias_sigma)
        
        # Calculate the KL divergence for each set of parameters
        kl_weight = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.Normal(self.weight_mu, self.weight_sigma),
            self.weight_prior))
        kl_bias = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.Normal(self.bias_mu, self.bias_sigma),
            self.bias_prior))
        
        return F.linear(input, weight, bias), kl_weight + kl_bias

    def kl_ib_output(self):
        batch_size = self.input.shape[0]
        input = self.input
        sig_weight = torch.exp(self.weight_rho)
        sig_bias = torch.exp(self.bias_rho)

        mu_out = F.linear(input, self.weight_mu, self.bias_mu)
        sig_out = F.linear(input.pow(2), sig_weight.pow(2), sig_bias.pow(2)).pow(0.5)
        kl_out = (- torch.log(sig_out) + 0.5 * (sig_out.pow(2) + mu_out.pow(2)) - 0.5).mean()
        if torch.isinf(kl_out):
            raise RuntimeError(kl_out, sig_out, mu_out)
        return kl_out

# class BayesianLinearPositive(nn.Module):
#     def __init__(self, input_features, output_features, prior_var=1.0):
#         super(BayesianLinearPositive, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
        
#         # Parameters for mean
#         self.weight_mu = nn.Parameter(torch.Tensor(output_features, input_features))
#         self.bias_mu = nn.Parameter(torch.Tensor(output_features))
        
#         # Parameters for variance (rho)
#         self.weight_rho = nn.Parameter(torch.Tensor(output_features, input_features))
#         self.bias_rho = nn.Parameter(torch.Tensor(output_features))
        
#         # Initialize parameters
#         self.reset_parameters()
        
#         # Prior distribution for all weights and biases
#         self.weight_prior = torch.distributions.Normal(0, prior_var)
#         self.bias_prior = torch.distributions.Normal(0, prior_var)
        
#         # Standard deviation of the prior
#         self.prior_var = prior_var

#     def reset_parameters(self):
#         # Initialize the parameters similar to a standard linear layer
#         nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.bias_mu, -bound, bound)
        
#         # Initialize rho parameters to provide a reasonable starting point
#         nn.init.constant_(self.weight_rho, -5)
#         nn.init.constant_(self.bias_rho, -5)

#     def forward(self, input, k=1):
#         self.input = input
#         # Calculate the standard deviation from rho
#         self.weight_sigma = torch.log1p(torch.exp(self.weight_rho)) #'weight_sigma'
#         # print(f"weight_sigma: {weight_sigma}")
#         self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
#         # Sample weights and biases from the posterior
#         # if setl.training:
#         weight = self.weight_mu + F.softplus(self.weight_sigma, beta=10) * torch.randn_like(self.weight_sigma)
#         bias = self.bias_mu + F.softplus(self.bias_sigma, beta=10) * torch.randn_like(self.bias_sigma)
        
#         # Calculate the KL divergence for each set of parameters
#         kl_weight = torch.sum(torch.distributions.kl_divergence(
#             torch.distributions.Normal(self.weight_mu, self.weight_sigma),
#             self.weight_prior))
#         kl_bias = torch.sum(torch.distributions.kl_divergence(
#             torch.distributions.Normal(self.bias_mu, self.bias_sigma),
#             self.bias_prior))
        
#         return F.linear(input, weight**2, bias**2), kl_weight + kl_bias

class BayesianLinearPositive(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.0):
        super(BayesianLinearPositive, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_mu = nn.Parameter(torch.Tensor(output_features))
        
        self.weight_rho = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_rho = nn.Parameter(torch.Tensor(output_features))
        
        self.reset_parameters()
        
        self.weight_prior = torch.distributions.Normal(0, prior_var)
        self.bias_prior = torch.distributions.Normal(0, prior_var)
        
        self.prior_var = prior_var

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        nn.init.constant_(self.weight_rho, -5)
        nn.init.constant_(self.bias_rho, -5)

    def forward(self, input, k=1):
        self.input = input
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho)).to(input.dtype)
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho)).to(input.dtype)
        
        weight = self.weight_mu.to(input.dtype) + F.softplus(self.weight_sigma, beta=10) * torch.randn_like(self.weight_sigma)
        bias = self.bias_mu.to(input.dtype) + F.softplus(self.bias_sigma, beta=10) * torch.randn_like(self.bias_sigma)
        
        kl_weight = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.Normal(self.weight_mu.to(input.dtype), self.weight_sigma),
            self.weight_prior))
        kl_bias = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.Normal(self.bias_mu.to(input.dtype), self.bias_sigma),
            self.bias_prior))
        
        return F.linear(input, weight**2, bias**2), kl_weight + kl_bias


    def kl_ib_output(self):
        batch_size = self.input.shape[0]
        input = self.input
        sig_weight = torch.exp(self.weight_rho)
        sig_bias = torch.exp(self.bias_rho)

        mu_out = F.linear(input, self.weight_mu, self.bias_mu)
        sig_out = F.linear(input.pow(2), sig_weight.pow(2), sig_bias.pow(2)).pow(0.5)
        kl_out = (- torch.log(sig_out) + 0.5 * (sig_out.pow(2) + mu_out.pow(2)) - 0.5).mean()
        if torch.isinf(kl_out):
            raise RuntimeError(kl_out, sig_out, mu_out)
        return kl_out



def concrete_sample(a, temperature, hard = False, eps = 1e-8, axis = -1, rand=True):
	"""
	Sample from the concrete relaxation.

	:param probs: torch tensor: probabilities of the concrete relaxation
	:param temperature: float: the temperature of the relaxation
	:param hard: boolean: flag to draw hard samples from the concrete distribution
	:param eps: float: eps to stabilize the computations
	:param axis: int: axis to perform the softmax of the gumbel-softmax trick

	:return: a sample from the concrete relaxation with given parameters
	"""

	device = torch.device("cuda" if a.is_cuda else "cpu")
	U = torch.rand(a.shape,device=device)
	G = - torch.log(- torch.log(U + eps) + eps)
	if rand==True:
		a=a*1.0

	t = (a + Variable(G)) / temperature

	y_soft = F.softmax(t, axis)

	if hard:
		#_, k = y_soft.data.max(axis)
		_, k = a.data.max(axis)
		shape = a.size()

		if len(probs.shape) == 2:
			y_hard = a.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
		else:
			y_hard = a.new(*shape).zero_().scatter_(-1, k.view(-1, probs.size(1), 1), 1.0)

		y = Variable(y_hard - y_soft.data) + y_soft
	else:
		y = y_soft
	return y
    
class BayesianBasicBlock(nn.Module):
    def __init__(self, input_size, hidden_size=32, U=2, temperature=0.67, prior_var=1.0, k_samples=1):
        super().__init__()
        
        self.U = U
        self.K = hidden_size // self.U
        self.n=0.0001 # To double check
        
        self.linear1 = BayesianLinear(input_size, hidden_size, prior_var=prior_var)
        # self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.temperature = temperature
        self.temp_test = 0.1 # 0.01

    def forward(self, x):
        kl_total = 0.
        
        out, kl = self.linear1(x)
        kl_total += kl
        # out = self.batchnorm1(out)
        # out, kl_lwta = self.lwta_activation(out, self.temperature if self.training else 0.1)
        out, kl_lwta = self.lwta_activation(out, temp=self.temperature, training=self.training)
        kl_total += kl_lwta
        return out, kl_total
        
    def lwta_activation(self, input, temp = 0.67, hard = False, training = True):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.
        # to try: logits = torch.reshape(input, [-1, input.size(ax)//U, U])
        # logits = torch.reshape(input, [-1, self.K, self.U])
        # logits = torch.reshape(input, [-1, input.size(-1), self.K, self.U]) # initial
        logits = torch.reshape(input, [-1, input.size(-1)//self.U, self.U])
       
        if training:   
            xi = concrete_sample(logits, temperature = temp, hard = hard, rand=True)
        else:
            xi = concrete_sample(logits, temperature = self.temp_test, hard = hard, rand=True)
        out = logits*xi
      
        out = out.reshape(input.shape)
    
        if self.training:
            q = F.softmax(logits, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q * (log_q - log_p), 1)
            # kl = torch.sum(kl) # initial
            kl = torch.mean(kl) / 1000.
            # self.n += len(q.view(-1)) # initial
            #scale up to be comparable with other terms
            # kl = kl * 100
        return out, kl
        
class BayesianSparseNeuralSurvivalTorch(nn.Module):
    def __init__(self, input_dim, layers, output_dim=1, U = 2, temperature = 0.67, prior_var=1.0, beta=1.0, alpha=1.0,
                 risks=1, dropout = 0., optimizer = "Adamw", seed=42):
        super(BayesianSparseNeuralSurvivalTorch, self).__init__()
        
        # Create layers for the initial parts of the network
        self.risks = risks
        self.beta = beta
        self.alpha = alpha
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.dropout = dropout
        self.seed = seed
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in layers:
            self.layers.append(
                BayesianBasicBlock(prev_dim, hidden_dim, U=U, temperature=temperature, prior_var=prior_var)
            )
            prev_dim = hidden_dim
        
        self.output_dim = output_dim
        self.last_hidden_dim = prev_dim
        self.predictor = BayesianLinearPositive(self.last_hidden_dim + 1, self.output_dim, prior_var=prior_var)
        # Need all values positive 
        self.softplus = nn.Sigmoid() #nn.Softplus()

    def forward(self, x, horizon):
        # Pass inputs through bayesian layers
        kl_total = 0.
        for layer in self.layers:
            x, kl = layer(x)
            kl_total += kl
        # information bound
        ib = self.layers[-1].linear1.kl_ib_output()
        
        # Compute cumulative hazard function
        tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
        output, kl_out = self.predictor(torch.cat((x, tau_outcome), 1))
        outcome = tau_outcome * self.softplus(output)
        # outcome = self.softplus(output)
        kl_total += kl_out
        
        self.kl_total = kl_total
        self.ib = ib
        
        return -outcome, tau_outcome, kl_total, ib
        
    def sample_elbo(self, x, horizon, samples=1):
        outputs = torch.zeros(samples, horizon.size(0), 1).to(x.device)
        kl_divergence = torch.zeros(samples).to(x.device)
        info_bound = torch.zeros(samples).to(x.device)
        
        for i in range(samples):
            outcome, tau_outcome, _, _ = self.forward(x, horizon)
            #print(f"outputs shape: {outputs.shape}, outcome shape: {outcome.shape}")
            outputs[i] = outcome#.squeeze()
            kl_divergence[i] = self.kl_total
            info_bound[i] = self.ib
        
        outputs = outputs.mean(0)
        kl_divergence = kl_divergence.mean()
        info_bound = info_bound.mean()

        return outputs, tau_outcome, kl_divergence, info_bound

    def gradient(self, outcomes, horizon, e):
        # Compute gradient for points with observed risk - Faster: only one backpropagation
        return grad([- outcomes[:, r][e == (r + 1)].sum() for r in range(self.risks)], horizon, create_graph = True)[0].clamp_(1e-10)[:, 0]
        
    # def predict_survival(self, x, t, risk=1):
    #     if not isinstance(t, list):
    #         t = [t]
    #     scores = []
    #     for t_ in t:
    #         t_ = torch.tensor([t_] * len(x), dtype=torch.float32, device=x.device)
    #         log_sr, _, _, _ = self.forward(x, t_)
    #         outcomes = 1 - (1 - torch.exp(log_sr))
    #         scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
    #     return np.concatenate(scores, axis=1)
        
