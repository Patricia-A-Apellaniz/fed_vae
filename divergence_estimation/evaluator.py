
import os
import sys
import torch

sys.path.insert(0, os.getcwd())

import numpy as np

from divergence_estimation.divergence import KL, JS


class DivergenceEvaluator:
    def __init__(self, x_p, x_q, id, result_path, pre_path, n, m, l, verbose=True, pr=None, ps=None, dataset_name=None):
        self.x_p = x_p
        self.x_q = x_q
        self.id = id
        self.result_path = result_path
        self.pre_path = pre_path
        self.n = n
        self.m = m
        self.l = l
        self.verbose = verbose
        self.pr = pr
        self.ps = ps
        self.dataset_name = dataset_name

    def evaluate(self, mc, ratio, disc, results_folder='results', cfg=None):
        # Compute divergence using different methods.
        M = self.m / (self.m + (2 * self.l)) # M is the fraction of the training set used to train the discriminator.
        n_fit_epochs = cfg.epochs # Number of epochs used to train the discriminator.

        # Define KL, JS object
        kl = KL(self.x_p, self.x_q, M, results_folder, seed=self.id, pre_path=self.pre_path, results_path=self.result_path, n=self.n,
                device=self.x_q.device, m=self.m, l=self.l, dataset_name=self.dataset_name, cfg=cfg)
        js = JS(self.x_p, self.x_q, M, results_folder, seed=self.id, pre_path=self.pre_path, results_path=self.result_path, n=self.n,
                device=self.x_q.device, m=self.m, l=self.l, dataset_name=self.dataset_name, cfg=cfg)
        kl1, kl2, kl4, js1, js2, js4 = None, None, None, None, None, None
        # Monte Carlo
        if mc:
            # check if pr is not none, else error or warning
            if self.pr is not None and self.ps is not None:
                kl1 = kl.mc(self.pr, self.ps)
                js1 = js.mc(self.pr, self.ps)
                if self.verbose:
                    print(f'KL divergence via Monte Carlo {np.round(kl1, 4)}')
                    print(f'JS divergence via Monte Carlo {np.round(js1, 4)}')
            else:
                print('ERROR: pr and ps are None, cannot compute MC')

        else:
            kl1 = torch.tensor(0)
            js1 = torch.tensor(0)

        # Perfect ratio
        if ratio:
            if self.pr is not None and self.ps is not None:
                kl2 = kl.perfect_ratio(self.pr, self.ps)
                js2 = js.perfect_ratio(self.pr, self.ps)
                if self.verbose:
                    print(f'KL divergence via r = p/q {np.round(kl2, 4)}')
                    print(f'JS divergence via r = p/q {np.round(js2, 4)}')
                else:
                    print('ERROR: pr and ps are None, cannot compute perfect ratio')
        else:
            kl2 = torch.tensor(0)
            js2 = torch.tensor(0)
        # Discriminator
        if disc:
            if self.result_path is None:
                path = None
            else:
                path = self.result_path + '/kl' + f'/{self.n}_{self.m}_{self.l}/seed_{self.id}_{self.dataset_name}_model.pt'
            kl4 = kl.forward(n_fit_epochs=n_fit_epochs, n=self.n, save_model=cfg.save_model, path=path, cfg=cfg)
            js4 = js.forward(n_fit_epochs=n_fit_epochs, n=self.n, cfg=cfg)
            kl4 = kl4.cpu().detach()
            js4 = js4.cpu().detach()
            if self.verbose:
                print(f'KL divergence via Discriminator {np.round(kl4.numpy(), 4)}')
                print(f'JS divergence via Discriminator {np.round(js4.numpy(), 4)}')
        else:
            kl4 = torch.tensor(0)
            js4 = torch.tensor(0)

        return kl1, kl2, kl4, js1, js2, js4


