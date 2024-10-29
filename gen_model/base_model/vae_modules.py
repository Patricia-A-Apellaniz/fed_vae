# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 14/06/2023

# Import libraries
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from .vae_utils import get_dim_from_type, get_activations_from_types, check_nan_inf


class LatentSpaceGaussian(object):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.latent_params = 2 * latent_dim  # Two parameters are needed for each Gaussian distribution

    def get_latent_params(self, x):
        x = x.view(-1, 2, self.latent_dim)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        return mu, log_var

    def sample_latent(self, latent_params):
        mu, log_var = latent_params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z

    def kl_loss(self, latent_params):
        mu, log_var = latent_params
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Kullback-Leibler divergence
        return kl / mu.shape[0]
        # TODO
        # return kl


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, grad_clip=1000.0, latent_limit=10.0):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.enc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # Note that this is a Gaussian encoder: latent_dim is the number of gaussian
        # distributions (hence, we need 2 * latent_dim parameters!)
        self.grad_clip = grad_clip
        self.latent_limit = latent_limit  # To limit the latent space values

    def forward(self, inp):
        x = self.enc1(inp)
        if x.requires_grad:
            x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
        x = F.relu(x)
        x = self.enc2(x)
        if x.requires_grad:
            x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
        x = torch.tanh(x) * self.latent_limit
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, feat_dists, max_k=10000.0, dropout_p=0.2, hidden_layers=2, hidden_size=50):
        super(Decoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.feat_dists = feat_dists
        self.out_dim = get_dim_from_type(self.feat_dists)
        if hidden_layers == 2:
            self.dec1 = nn.Linear(in_features=latent_dim, out_features=hidden_size)
            self.dec2 = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
        elif hidden_layers == 3:
            self.dec1 = nn.Linear(in_features=latent_dim, out_features=hidden_size + 25)
            self.dec2 = nn.Linear(in_features=hidden_size + 25, out_features=hidden_size)
            self.dec3 = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
        self.max_k = max_k
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z):
        x = F.relu(self.dec1(z))
        if self.hidden_layers > 1:
            x = self.dropout(x)
            x = self.dec2(x)
        if self.hidden_layers == 3:
            x = self.dropout(x)
            x = self.dec3(x)
        x = get_activations_from_types(x, self.feat_dists, max_k=self.max_k)
        return x


class LogLikelihoodLoss(nn.Module):

    def __init__(self, feat_dists):
        super(LogLikelihoodLoss, self).__init__()
        self.feat_dists = feat_dists

    def forward(self, inputs, targets, imp_mask):
        index_x = 0  # Index of parameters (do not confound with index_type below, which is the index of the
        # distributions!)
        loss_ll = []  # Loss for each distribution will be appended here
        # Append covariates losses
        for index_type, type in enumerate(self.feat_dists):
            if type[0] == 'negative-binomial':
                dist, num_params, r = type
            else:
                dist, num_params = type
            if dist == 'gaussian' or dist == 'log-normal':
                mean = inputs[:, index_x]
                std = inputs[:, index_x + 1]
                ll = - torch.log(np.sqrt(2 * np.pi) * std) - 0.5 * ((targets[:, index_type] - mean) / std).pow(2)
            elif dist == 'bernoulli':
                p = inputs[:, index_x]
                ll = targets[:, index_type] * torch.log(p) + (1 - targets[:, index_type]) * torch.log(1 - p)
            elif dist == 'categorical':
                p = inputs[:, index_x: index_x + num_params]
                mask = F.one_hot(targets[:, index_type].long(), num_classes=num_params)  # These are the indexes
                # whose losses we want to compute
                ll = torch.log(p) * mask
            elif dist == 'negative-binomial':
                r = inputs[:, index_x]
                p = inputs[:, index_x + 1]
                ll = torch.lgamma(targets[:, index_type] + r) - torch.lgamma(r) - torch.lgamma(
                    targets[:, index_type] + 1) + targets[:, index_type] * torch.log(1 - p) + r * torch.log(p)
            elif dist == 'poisson':
                lam = inputs[:, index_x]
                ll = targets[:, index_type] * torch.log(lam) - lam - torch.lgamma(targets[:, index_type] + 1)
            elif dist == 'exponential':
                lam = inputs[:, index_x]
                ll = torch.log(1/lam) - (targets[:, index_type] / lam)
            elif dist == 'weibull':
                alpha = inputs[:, index_x]
                lam = inputs[:, index_x + 1]
                ll = torch.log(alpha / lam) + (alpha - 1) * torch.log(targets[:, index_type] / lam) - (
                        targets[:, index_type] / lam).pow(alpha)
            elif dist == 'beta':
                alpha = inputs[:, index_x]
                beta = inputs[:, index_x + 1]
                # Note that we limit the beta values to be in the range (0, 1)
                ll = (alpha - 1) * torch.log(targets[:, index_type]) \
                     + (beta - 1) * torch.log(1 - (targets[:, index_type])) \
                     + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
            else:
                raise RuntimeError('Unknown distribution to compute loss')
            check_nan_inf(ll, 'Covariates loss')
            if 0 in imp_mask:
                if dist == 'categorical':
                    ll *= imp_mask[:, index_type].unsqueeze(1)
                else:
                    ll *= imp_mask[:, index_type]  # Add the imputation effect: do NOT train on outputs with mask=0!
            loss_ll.append(-torch.sum(ll) / inputs.shape[0])
            index_x += num_params

        return sum(loss_ll)


class LogLikelihoodLossWithCensoring(nn.Module):

    def __init__(self, dist_type):
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.type = dist_type

    def forward(self, inputs, targets, risk):
        targets = torch.squeeze(targets)
        # Append time losses, follow Liverani paper to account for censorship: hazard is only used when no censoring!
        risk = np.squeeze(risk)
        if self.type[0] == 'weibull':
            alpha = inputs[:, 0]
            lam = inputs[:, 1]
            surv = - (targets / lam) ** alpha  # Always present
            # hazard = torch.log(alpha / lam) + (alpha - 1) * torch.log(
            #     targets / lam)  # Present only when not censored data!
            hazard = torch.log(alpha / lam) + (alpha - 1) * torch.log(
                targets / lam) - ((targets / lam) ** alpha)  # Present only when not censored data!

            loss_ll = - torch.sum(hazard[risk == 1]) - torch.sum(surv)
            check_nan_inf(surv, 'Time loss: surv')
            check_nan_inf(hazard, 'Time loss: hazard')

        elif self.type[0] == 'exponential':
            lam = inputs[:, 0]
            surv = - (targets / lam)  # Always present
            hazard = torch.log(1/lam) - (targets / lam)  # Present only when not censored data!
            loss_ll = - torch.sum(hazard[risk == 1]) - torch.sum(surv)
            check_nan_inf(surv, 'Time loss: surv')
            check_nan_inf(hazard, 'Time loss: hazard')

        elif self.type[0] == 'poisson':
            lam = inputs[:, 0]
            # import torch.distributions as dist
            # poisson_dist = dist.Poisson(lam)
            # surv = 1 - poisson_dist.cdf(targets)
            surv = torch.zeros_like(targets, dtype=torch.float32)
            for i in range(targets.size(0)):
                sum_prob = 0.0
                for j in range(int(targets[i]) + 1):
                    # Calcular la probabilidad de que X sea menor o igual a k[i]
                    prob = 1 - (- lam[i] + j * torch.log(lam[i]) - torch.lgamma(torch.tensor(j) + 1))
                    sum_prob += prob

                surv[i] = sum_prob

            loglikelihood = targets * torch.log(lam) - lam - torch.lgamma(targets + 1)
            loss_ll = -torch.sum(torch.where(risk == 1, loglikelihood, surv))
            check_nan_inf(surv, 'Time loss: surv')


        elif self.type[0] == 'gaussian':
            from scipy.special import erf
            mean = inputs[:, 0]
            std = inputs[:, 1]
            # surv = torch.tensor(0.5 * (1 + torch.erf(targets - mean / (std * np.sqrt(2.0)))))
            # hazard = - torch.log(std * np.sqrt(2 * np.pi)) - (targets - mean).pow(2) / (2 * std.pow(2)) - np.log(0.5) + torch.log(1 - erf(targets.numpy() - mean.detach().numpy() / (std.detach().numpy() * np.sqrt(2.0)))))

            surv = 0.5 * (1 + torch.erf((targets - mean) / (std * np.sqrt(2)))) * (1.0 - 2 * 1e-6) + 1e-6
            loglikelihood = -0.5 * torch.log(2 * np.pi * std ** 2) - (targets - mean) ** 2 / (2 * std ** 2)
            loss_ll = -torch.sum(torch.where(risk == 1, loglikelihood, surv))
            check_nan_inf(surv, 'Time loss: surv')
            check_nan_inf(loglikelihood, 'Time loss: hazard')

        else:
            raise RuntimeError('Unknown time distribution to compute loss')
        return loss_ll / inputs.shape[0]


# In this case, there is no liverani ??
class LogLikelihoodLossWithCensoring_t_e(nn.Module):

    def __init__(self, dist_types):
        super(LogLikelihoodLossWithCensoring_t_e, self).__init__()
        self.sa_dists = [dist_types] + [('bernoulli', 1)]

    def forward(self, inputs, targets, risks):
        # Concatenate targets and risks
        targets = torch.unsqueeze(targets, dim=1)
        risks = torch.unsqueeze(risks, dim=1)
        targets = torch.cat((targets, risks), dim=1)
        index_x = 0  # Index of parameters (do not confound with index_type below, which is the index of the
        # distributions!)
        loss_ll = []  # Loss for each distribution will be appended here
        # Append covariates losses
        for index_type, type in enumerate(self.sa_dists):
            dist, num_params = type
            if dist == 'gaussian' or dist == 'log-normal':
                mean = inputs[:, index_x]
                std = inputs[:, index_x + 1]
                ll = - torch.log(np.sqrt(2 * np.pi) * std) - 0.5 * ((targets[:, index_type] - mean) / std).pow(2)
            elif dist == 'bernoulli':
                p = inputs[:, index_x]
                ll = targets[:, index_type] * torch.log(p) + (1 - targets[:, index_type]) * torch.log(1 - p)
            elif dist == 'categorical':
                p = inputs[:, index_x: index_x + num_params]
                mask = F.one_hot(targets[:, index_type].long(), num_classes=num_params)  # These are the indexes
                # whose losses we want to compute
                ll = torch.log(p) * mask
            elif dist == 'weibull':
                alpha = inputs[:, index_x]
                lam = inputs[:, index_x + 1]
                ll = torch.log(alpha / lam) + (alpha - 1) * torch.log(targets[:, index_type] / lam) - (
                        targets[:, index_type] / lam).pow(alpha)
            else:
                raise RuntimeError('Unknown distribution to compute time loss')
            check_nan_inf(ll, 'Time loss')
            loss_ll.append(-torch.sum(ll) / inputs.shape[0])
            index_x += num_params

        return sum(loss_ll)
