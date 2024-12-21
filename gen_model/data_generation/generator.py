# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 12/09/2023


# Packages to import
import torch

import numpy as np

from sklearn.mixture import BayesianGaussianMixture
from gen_model.base_model.vae_model import VariationalAutoencoder
from gen_model.base_model.vae_utils import check_nan_inf, sample_from_dist


class Generator(VariationalAutoencoder):
    """
    Module implementing Synthethic data generator
    """

    def __init__(self, params):
        # Initialize Generator parameters and modules
        super(Generator, self).__init__(params)
        self.bgm = None

    def train_latent_generator(self, x, device):  # Train latent generator using x data as input
        if self.bgm is not None:
            raise RuntimeWarning(['WARNING] BGM is being retrained'])

        vae_data = self.predict(x, device=device)
        mu_latent_param, log_var_latent_param = vae_data['latent_params']

        # Fit GMM to the mean
        converged = False
        bgm = None
        n_try = 0
        max_try = 100
        # NOTE: this code is for Gaussian latent space, change it if using a different one!
        while not converged and n_try < max_try:  # BGM may not converge: try different times until it converges
            # (or it reaches a max number of iterations)
            n_try += 1
            n_try += 1
            bgm = BayesianGaussianMixture(n_components=self.latent_dim, random_state=42 + n_try, reg_covar=1e-5,
                                          n_init=10, max_iter=5000).fit(mu_latent_param)  # Use only mean
            converged = bgm.converged_

        if not converged:
            print('[WARNING] BGM did not converge after ' + str(n_try + 1) + ' attempts')
            print('NOT CONVERGED')
        else:
            self.bgm = {'bgm': bgm,
                        'log_var_mean': np.mean(log_var_latent_param, axis=0)}  # BGM data to generate patients

    def generate(self, n_gen=100, device=None):
        if self.bgm is None:
            print('[WARNING] BGM  is not trained, try calling train_latent_generator before calling generate')
        else:
            mu_sample = self.bgm['bgm'].sample(n_gen)[0]
            log_var_sample = np.tile(self.bgm['log_var_mean'], (n_gen, 1))

            z = self.latent_space.sample_latent([torch.from_numpy(mu_sample).float(),
                                                 torch.from_numpy(log_var_sample).float()]).to(device)
            check_nan_inf(z, 'GMM latent space')
            cov_params = self.Decoder(z)
            check_nan_inf(cov_params, 'Decoder')
            cov_params = cov_params.detach().cpu().numpy()
            cov_samples = sample_from_dist(cov_params, self.feat_distributions)
            out_data = {'z': z.detach().cpu().numpy(),
                        'cov_params': cov_params,
                        'cov_samples': cov_samples,
                        'latent_params': [mu_sample, log_var_sample]}

            return out_data



    def tvae_generator(self, n_gen=100, device=None):
        mu_sample = np.zeros((n_gen, self.latent_dim))
        log_var_sample = np.zeros((n_gen, self.latent_dim))
        z = self.latent_space.sample_latent([torch.from_numpy(mu_sample).float(), torch.from_numpy(log_var_sample).float()]).to(device)
        check_nan_inf(z, 'TVAE latent space')

        # Sample from z
        cov_params = self.Decoder(z)
        check_nan_inf(cov_params, 'Decoder')
        cov_params = cov_params.detach().cpu().numpy()
        cov_samples = sample_from_dist(cov_params, self.feat_distributions)
        out_data = {'z': z.detach().cpu().numpy(),
                    'cov_params': cov_params,
                    'cov_samples': cov_samples,
                    'latent_params': [mu_sample, log_var_sample]}

        return out_data