# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/10/2024

# Importing libraries
import os
import time
import torch
import argparse

import numpy as np
import pandas as pd

from gen_model.data import split_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from gen_model.data_generation.generator import Generator
from divergence_estimation.evaluator import DivergenceEvaluator



class Node(object):
    def __init__(self, df_train, df_val, feat_distributions, n_seeds, latent_dim=10, hidden_size=256, dataset_name='4', node_name='0'):
        self.private_df = df_train  # private data used for training
        self.val_df = df_val  # validation data (also private)
        self.public_df = None
        self.shared_df = None
        self.params = {'feat_distributions': feat_distributions, 'latent_dim': latent_dim, 'hidden_size': hidden_size,
                       'input_dim': self.private_df.shape[1]}  # parameters for the generator
        self.models = [Generator(self.params) for _ in range(n_seeds)]  # models for the generator: one per seed
        self.best_model = None
        self.dataset_name = dataset_name
        self.node_name = node_name
        self.model_public_df = None

    def train_on_local_data(self, n_epochs=10000, batch_size=64, lr=1e-3, use_shared_data=False, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), max_samples_train=None):
        train_params = {'n_epochs': n_epochs, 'batch_size': batch_size, 'device': device, 'lr': lr}
        # Training data: private data + shared data
        if use_shared_data:
            train_data = pd.concat([self.private_df, self.shared_df], axis=0)
        else:
            train_data = self.private_df
        if max_samples_train is not None and train_data.shape[0] > max_samples_train:
            train_data = train_data[:max_samples_train]  # Always use the private samples

        mask = pd.DataFrame(np.ones(train_data.shape), columns=train_data.columns)
        self.data = split_data(train_data, mask)

        best_loss = float('inf')
        training_results = []
        for i, model in enumerate(self.models):
            mask = pd.DataFrame(np.ones(train_data.shape), columns=train_data.columns)
            self.data = split_data(train_data, mask)

            print(f"Training model {i}/{len(self.models)} on node {self.node_name} with {train_data.shape[0]} data points")
            training_results.append(model.fit(self.data, train_params))
            if training_results[-1]['loss_va'][-1] < best_loss:
                best_loss = training_results[-1]['loss_va'][-1]
                self.best_model = i
            print(f"Training model {i}/{len(self.models)} on node {self.node_name} finished")

        return training_results

    def generate_public_data(self, n_samples=1000, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), max_bgm_points=None):
        # Generate shared data
        print(f"Node {self.node_name}: Generating {n_samples} samples")
        public_data = []

        if max_bgm_points is not None and self.data[0].shape[0] > max_bgm_points:
            bgm_data = self.data[0][0: max_bgm_points]
        else:
            bgm_data = self.data[0]

        for i, model in enumerate(self.models): # By default, generate using all the models
            print(f"Training BGM model...")
            t0 = time.time()
            model.bgm = None  # Reset the BGM model
            model.train_latent_generator(bgm_data, device)  # Train the latent generator, let it ready for data generation afterwards
            print( f"Training model {i}/{len(self.models)} on node {self.node_name} finished: BGM model trained in {time.time() - t0} seconds")
            # Generate data
            gen_data = model.generate(n_gen=n_samples, device=device)
            gen_df = pd.DataFrame(gen_data['cov_samples'], columns=self.private_df.columns.tolist())
            public_data.append(gen_df)
        self.model_public_df = public_data
        self.public_df = pd.concat(public_data, axis=0)
        self.public_df = self.public_df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
        return self.public_df

    def add_shared_data(self, shared_df, n=2000):
        print(f"Node {self.node_name}: Received {shared_df.shape[0]} samples from other nodes")
        self.shared_df = shared_df.sample(frac=1).reset_index(drop=True)  # Shuffle the data received from other nodes
        print(f"Node {self.node_name}: Received {shared_df.shape[0]} samples from other nodes")

def evaluate_node(dataset_name, dir_name, exp_type, node_name, round, m, l, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), clas=False, label=None, seed=0, results_path='results'):  # Evaluate the quality of the generated data using the JS divergence
    syn_df = pd.read_csv(os.path.join(results_path, dir_name, 'public_data_' + exp_type + '_r_' + str(round) + '_n_' + node_name + '.csv'))
    val_df = pd.read_csv(os.path.join(results_path, dir_name, 'val_data_' + node_name + '.csv'))

    n = val_df.shape[0]
    val_size = m + 2 * l

    print(f"Node {node_name}: Evaluating on {val_size} samples")

    x_real = torch.tensor(val_df.values[0:val_size], dtype=torch.float32, device=device)
    x_gen = torch.tensor(syn_df.values[0: val_size], dtype=torch.float32, device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10000, type=int, help='Number of epochs to train the discriminator')
    parser.add_argument('--save_model', default=False, type=bool, help='Save the model')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--use_pretrained', default=False, type=bool, help='Whether to use a pretrained model')
    parser.add_argument('--print_feat_js', default=False, type=bool, help='Whether to print the JS per feature')
    parser.add_argument('--name', default=dataset_name, type=str, help='Name of the data')

    cfg = parser.parse_args()
    evaluator = DivergenceEvaluator(x_real, x_gen, id=seed, result_path=None,
                                    pre_path=None, n=n, m=m, l=l, verbose=True, pr=None, ps=None,
                                    dataset_name=dataset_name)
    _, _, kl, _, _, js = evaluator.evaluate(mc=False, ratio=False, disc=True, cfg=cfg)
    kl = kl.item()
    js = js.item()

    if clas:  # Add a classifier to compare
        y_real = val_df[label].values[0:val_size]
        x_real = val_df.drop(columns=label).values[0:val_size]
        y_gen = syn_df[label].values[0:val_size]
        x_gen = syn_df.drop(columns=label).values[0:val_size]

        split = 0.8
        x_train_real = x_real[0:int(split * val_size)]
        y_train_real = y_real[0:int(split * val_size)]
        x_test_real = x_real[int(split * val_size):]
        y_test_real = y_real[int(split * val_size):]
        x_train_gen = x_gen[0:int(split * val_size)]
        y_train_gen = y_gen[0:int(split * val_size)]
        x_test_gen = x_gen[int(split * val_size):]
        y_test_gen = y_gen[int(split * val_size):]

        model_real = RandomForestClassifier(n_estimators=100)
        model_real.fit(x_train_real, y_train_real)
        acc_real_real = accuracy_score(y_test_real, model_real.predict(x_test_real))
        acc_real_gen = accuracy_score(y_test_gen, model_real.predict(x_test_gen))

        model_gen = RandomForestClassifier(n_estimators=100)
        model_gen.fit(x_train_gen, y_train_gen)
        acc_gen_real = accuracy_score(y_test_real, model_gen.predict(x_test_real))
        acc_gen_gen = accuracy_score(y_test_gen, model_gen.predict(x_test_gen))

        pd.DataFrame({'kl': [kl], 'js': [js], 'acc_real_real': [acc_real_real], 'acc_real_gen': [acc_real_gen],
                      'acc_gen_real': [acc_gen_real], 'acc_gen_gen': [acc_gen_gen], 'n_val': [len(x_test_real)]}).to_csv(
            os.path.join(results_path, dir_name, 'evaluation_' + exp_type + '_r_' + str(round) + '_n_' + node_name + '_s_' + str(seed) + '.csv'))
    else:
        pd.DataFrame({'kl': [kl], 'js': [js]}).to_csv(
            os.path.join(results_path, dir_name, 'evaluation_' + exp_type + '_r_' + str(round) + '_n_' + node_name + '_s_' + str(seed) +  '.csv'))


