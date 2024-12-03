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
from distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma
# from torch.distributions.Pa

class BayesianLinear(nn.Module):
    """
    Single linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config,
            use_mixture: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights
        
        self.config = config
        if self.config['mu_scale'] is None:
            self.weight_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        else:
            self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(out_features, in_features)),
                                              -self.config['mu_scale'], self.config['mu_scale'])

        self.weight_rho = nn.Parameter(torch.ones([out_features, in_features]) * self.config['rho_scale'])
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * self.config['rho_scale'])
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if use_mixture:
            pi = config['pi']
        else:
            pi = 1
        self.weight_prior = ScaleMixtureGaussian(pi, config['sigma1'], config['sigma2'])
        self.bias_prior = ScaleMixtureGaussian(pi, config['sigma1'], config['sigma2'])

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1,
            positive: bool = False
    ):
        self.input = x
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
            bias = self.bias.sample(n_samples=n_samples)
        else:
            # print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(torch.nan_to_num(weight, nan=1e-6)) + self.bias_prior.log_prob(torch.nan_to_num(bias, nan=1e-6))
            self.log_variational_posterior = self.weight.log_prob(torch.nan_to_num(weight, nan=1e-6)) + self.bias.log_prob(torch.nan_to_num(bias, nan=1e-6))
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # For a single layer network, x would have 2 dimension [n_data, n_feature]
        # But sometime x would be the sampled output from the previous layer,
        # which will have 3 dimension [n_samples, n_data, n_feature]
        n_data = x.shape[-2]
        bias = bias.repeat(1, n_data, 1)
        # If x is 3-d, this expand command will make x remains the same.
        x = x.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: input features size; k: output size
        if positive:
            return torch.einsum('bij,bkj->bik', x, weight ** 2) + bias ** 2
        return torch.einsum('bij,bkj->bik', x, weight) + bias

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

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, self.config['rho_scale'])
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config['rho_scale'])
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

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
    def __init__(self, input_size, hidden_size, config):
        super().__init__()

        self.config = config
        self.U = config['U']
        self.K = hidden_size // self.U
        # self.n=0.0001 # To double check
        
        self.linear1 = BayesianLinear(input_size, hidden_size, config)
        self.temperature = config['temperature']
        self.temp_test = config['temp_test'] # 0.01
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x, sample=False, n_samples=1):
        out = self.linear1(x, sample=sample, n_samples=n_samples)
        out, kl_lwta = self.lwta_activation(out, temp=self.temperature, training=self.training)
        if self.training:
            out = self.dropout(out)
        self.kl_lwta = kl_lwta
        return out.mean(dim=0)
        
    def loss(self):
        return (self.linear1.log_variational_posterior - self.linear1.log_prior) + self.kl_lwta
        
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
    def __init__(self, input_dim, layers, config, output_dim=1, seed=42):
        super(BayesianSparseNeuralSurvivalTorch, self).__init__()
        
        # Create layers for the initial parts of the network
        self.risks = config['risks']
        self.beta = config['beta']
        self.alpha = config['alpha']
        self.optimizer = config['optimizer']
        self.input_dim = input_dim
        self.dropout = config['dropout']
        self.seed = config['seed']
        self.config = config
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in layers:
            self.layers.append(
                BayesianBasicBlock(prev_dim, hidden_dim, config)
            )
            prev_dim = hidden_dim
        
        self.output_dim = output_dim
        self.last_hidden_dim = prev_dim
        self.predictor = BayesianLinear(self.last_hidden_dim + 1, self.output_dim, config)
        # Need all values positive 
        self.softplus = nn.Softplus() # nn.Sigmoid() 

    def forward(self, x, horizon, sample=False, n_samples=1):
        # Pass inputs through bayesian layers
        kl_total = 0.
        for layer in self.layers:
            x = layer(x, sample=sample, n_samples=n_samples)
            kl_total += layer.loss()
        # information bound
        ib = self.layers[-1].linear1.kl_ib_output()
        
        # Compute cumulative hazard function
        tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
        output = self.predictor(torch.cat((x, tau_outcome), 1), sample=sample, n_samples=n_samples, positive=True)
        output = output.mean(0)
        outcome = tau_outcome * self.softplus(output)
        # outcome = self.softplus(output)
        kl_total += self.predictor.log_variational_posterior - self.predictor.log_prior
        
        self.kl_total = kl_total
        self.ib = ib
        
        return -outcome, tau_outcome, kl_total, ib
        
    def sample_elbo(self, x, horizon):
        n_samples = self.config['n_samples_train']
        outputs, tau_outcome, _, _ = self.forward(x, horizon, sample=True, n_samples=n_samples)
        outputs = outputs.mean(0)
        kl_divergence = self.kl_total / n_samples
        info_bound = self.ib / n_samples

        return outputs, tau_outcome, kl_divergence, info_bound 

    def gradient(self, outcomes, horizon, e):
        # Compute gradient for points with observed risk - Faster: only one backpropagation
        return grad([- outcomes[:, r][e == (r + 1)].sum() for r in range(self.risks)], horizon, create_graph = True)[0].clamp_(1e-10)[:, 0]