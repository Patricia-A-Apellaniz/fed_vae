# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/03/2023

# Import libraries
import math
import torch

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------
#                      TRAINING PROCESS
# -----------------------------------------------------------


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

    def reset(self):
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False


# TODO: remove unused distributions
def sample_from_dist(params, feat_dists, mode='sample'):  # Get samples from the base_model
    i = 0
    out_vals = []
    for type in feat_dists:
        if type[0] == 'negative-binomial':
            dist, num_params, r = type
        else:
            dist, num_params = type
        if dist == 'gaussian' or dist == 'log-normal':
            if mode == 'sample':
                x = np.random.normal(loc=params[:, i], scale=params[:, i + 1])
                if dist == 'gaussian':
                    out_vals.append(x)
                elif dist == 'log-normal':
                    out_vals.append(np.exp(x))
            elif mode == 'mode' or mode == 'mean':
                out_vals.append(params[:, 1])  # Mean / mode of the distribution
        elif dist == 'weibull':
            if mode == 'sample':
                # out_vals.append(np.random.weibull(a=params[:, i]) * params[:, i + 1])
                out_vals.append(np.random.weibull(a=params[:, i]) * params[:, i + 1])
            elif mode == 'mode':
                a = params[:, i]
                lam = params[:, i + 1]
                mode = lam * np.power((a - 1) / a, 1.0 / a)  # Mode of the Weibull distribution
                mode[a <= 1] = 0  # The mode of the weibull has two possible values
                out_vals.append(mode)
            elif mode == 'mean':
                out_vals.append(params[:, i + 1] * math.gamma(1 + 1.0 / params[:, i]))  # Mean of the distribution
        elif dist == 'exponential':
            if mode == 'sample':
                out_vals.append(np.random.exponential(scale=params[:, i]))
            elif mode == 'mode':
                pass
            elif mode == 'mean':
                pass
        elif dist == 'bernoulli':
            if mode == 'sample':
                out_vals.append(np.random.binomial(n=np.ones_like(params[:, i]).astype(int), p=params[:, i]))
            elif mode == 'mode':
                out_vals.append((params[:, 1] > 0.5).astype(int))
            elif mode == 'mean':
                out_vals.append(params[:, 1])
        # elif dist == 'beta':
        #     if mode == 'sample':
        #         out_vals.append(np.random.beta(a=params[:, i], b=params[:, i + 1]))
        #     elif mode == 'mode' or mode == 'mean':
        #         raise NotImplementedError
        # elif dist == 'negative-binomial':
        #     if mode == 'sample':
        #         out_vals.append(np.random.negative_binomial(n=params[:, i], p=params[:, i + 1]))
        #     elif mode == 'mode' or mode == 'mean':
        #         raise NotImplementedError
        elif dist == 'poisson':
            if mode == 'sample':
                out_vals.append(np.random.poisson(lam=params[:, i]))
            elif mode == 'mode' or mode == 'mean':
                raise NotImplementedError
        elif dist == 'categorical':
            if mode == 'sample':
                aux = np.zeros((params.shape[0],))
                for j in range(params.shape[0]):
                    aux[j] = np.random.choice(np.arange(num_params), p=params[j, i: i + num_params])  # Choice
                    # takes p as vector only: we must proceed one by one
                out_vals.append(aux)
            elif mode == 'mode':
                raise NotImplementedError
            elif mode == 'mean':
                raise NotImplementedError
        i += num_params
    return np.array(out_vals).T


def plot_losses(train_loss, val_loss, title, fig_path):
    plt.figure(figsize=(15, 15))
    plt.semilogy(train_loss, label='Train')
    plt.semilogy(val_loss, label='Valid')
    plt.title('Train and Validation ' + title)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.savefig(fig_path)
    # plt.show()
    plt.close()


def check_nan_inf(values, log):
    if torch.isnan(values).any().detach().cpu().tolist() or torch.isinf(values).any().detach().cpu().tolist():
        print('[WARNING] NAN DETECTED. ' + str(log))
    return


def get_dim_from_type(feat_dists):
    return sum(d[1] for d in feat_dists)  # Returns the number of parameters needed
    # to base_model the distributions in feat_dists


def get_activations_from_types(x, feat_dists, min_val=1e-3, max_std=10.0, max_alpha=2, max_k=1000.0):
    # Ancillary function that gives the correct torch activations for each data distribution type
    # Example of type list: [('bernoulli', 1), ('gaussian', 2), ('categorical', 5)]
    # (distribution, number of parameters needed for it)
    index_x = 0
    out = []
    for index_type, type in enumerate(feat_dists):
        if type[0] == 'negative-binomial':
            dist, num_params, r = type
        else:
            dist, num_params = type
        if dist == 'gaussian':
            out.append(torch.tanh(x[:, index_x, np.newaxis]) * 5)  # Mean: from -inf to +inf
            out.append((torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / (10 * max_std))
        elif dist == 'log-normal':
            out.append(x[:, index_x, np.newaxis])  # Mean: from -inf to +inf
            # Std: from 0 to +inf
            out.append((torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / (10 * max_std))
        elif dist == 'exponential':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (max_k[-1] - min_val) + min_val)
            #out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (0.5 - min_val) + min_val)
            #out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (max_k[-1]/2 - min_val) + min_val)
        elif dist == 'weibull':
            out.append((torch.sigmoid(x[:, index_x, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_alpha)
            out.append(
                (torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_k)
            # K: (min_val, max_k + min_val)
        elif dist == 'bernoulli':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (1.0 - 2 * min_val) + min_val)
            # p: (min_val, 1-min_val)
        elif dist == 'poisson':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (10 - 2 * min_val) + min_val)
        elif dist == 'beta':  # Beta params are positive!
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (1000.0 - min_val) + min_val)
            out.append(torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (1000.0 - min_val) + min_val)
        elif dist == 'categorical':  # Softmax activation: NANs appear if values are close to 0,
            # so use min_val to prevent that
            vals = torch.tanh(x[:, index_x: index_x + num_params]) * 10.0  # Limit the max values
            out.append(torch.softmax(vals, dim=1))  # probability of each categorical value # TODO: check vals type
            check_nan_inf(out[-1], 'Categorical distribution')
        elif dist == 'negative-binomial':
            out.append(torch.tanh(x[:, index_x, np.newaxis]))  # R > 0
            out.append(torch.tanh(x[:, index_x + 1, np.newaxis]))  # p: [0,1]
        else:
            raise NotImplementedError('Distribution ' + dist + ' not implemented')
        index_x += num_params
    return torch.cat(out, dim=1)


def linear_rate(epoch, n_epochs, ann_prop):
    # Adjust the KL parameter with a constant annealing rate
    if epoch >= n_epochs * ann_prop - 1:
        factor = 1
    else:
        factor = 1 / (ann_prop * n_epochs) * epoch  # Linear increase
    return factor


def cyclic_rate(epoch, n_epochs, n_cycles, ann_prop=0.5):
    # Based on the paper: Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
    epochs_per_cycle = int(np.ceil(n_epochs / n_cycles))
    return linear_rate(epoch % epochs_per_cycle, epochs_per_cycle, ann_prop)


def triangle_rate(epoch, n_epochs_total, n_epochs_init, init_val, ann_prop):
    if epoch <= n_epochs_init:
        return init_val
    else:
        return linear_rate(epoch - n_epochs_init, n_epochs_total - n_epochs_init, ann_prop)
