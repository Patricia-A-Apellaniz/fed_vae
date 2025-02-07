# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/12/2024

# Importing libraries
import os
import torch
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from data_utils import load_data


if __name__ == '__main__':
    dataset_names = ['3', '7']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    concern_1 = True
    concern_3 = True
    concern_5 = True
    results_path = 'results'

    # STEPS (CONCERNS 1 AND 6)
    if concern_1:
        n_nodes = 3
        m = 7500
        l = 1000
        exp_type = 'drs'
        node_name = '2'
        round = 4
        calculate_distances = False

        for dataset_name in dataset_names:
            for i, iid in enumerate([True, False]):
                if iid:
                    dir_name = dataset_name + '_iid'
                else:
                    dir_name = dataset_name + '_niid'

                # 1. Load real and synthetic data
                syn_df = pd.read_csv(os.path.join(results_path, dir_name, 'public_data_' + exp_type + '_r_' + str(round) + '_n_' + node_name + '.csv'))
                real_df = pd.read_csv(os.path.join(results_path, dir_name, 'val_data_' + node_name + '.csv'))
                val_size = m + 2 * l
                x_real = real_df.values[0:val_size]
                x_gen = syn_df.values[0: val_size]

                if calculate_distances:
                    distances = {}

                    # 2. Take real data and compare each sample with the rest checking distances
                    # Obtain the minimum distance from each point to all the real data (one to all)
                    real_dists = np.ones((x_real.shape[0], x_real.shape[0])) * np.inf
                    for i in range(x_real.shape[0]):  # Note: this may not be the most efficient implementation...
                        for j in range(x_real.shape[0]):
                            if i != j:
                                real_dists[i, j] = np.linalg.norm(x_real[i] - x_real[j])
                    # Save distances
                    distances['real'] = real_dists

                    # 3. Take synthetic data and compare each sample with the rest real samples checking distances
                    # Obtain the minimum distance from each point to all the real data (one to all)
                    gen_dists = np.ones((x_gen.shape[0], x_real.shape[0])) * np.inf
                    for i in range(x_gen.shape[0]):  # Note: this may not be the most efficient implementation...
                        for j in range(x_real.shape[0]):
                            if i != j:
                                gen_dists[i, j] = np.linalg.norm(x_gen[i] - x_real[j])
                    # Save distances
                    distances['gen'] = gen_dists

                    # 4. Save dictionary with distances
                    with open(os.path.join(results_path, dir_name) + os.sep + exp_type + '_distances.pickle', 'wb') as handle:
                        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    # Load dictionary with distances
                    with open(os.path.join(results_path, dir_name) + os.sep + exp_type + '_distances.pickle', 'rb') as handle:
                        distances = pickle.load(handle)

                # 4. Obtain minimum distances (one to one)
                min_real_dists = np.min(distances['real'], axis=0)
                min_gen_dists = np.min(distances['gen'], axis=0)

                # 4. Generate histograms with minimum distances for each one
                fig, ax = plt.subplots(figsize = (16, 12))
                plt.tight_layout()
                min_real_dists_df = pd.DataFrame(min_real_dists)
                min_real_dists_df.plot(ax=ax, kind='hist', density=True, alpha=0.8, color='skyblue',bins=30)
                min_real_dists_df.plot(ax=ax, kind='kde', color='blue', bw_method=0.3)
                min_gen_dists_df = pd.DataFrame(min_gen_dists)
                min_gen_dists_df.plot(ax=ax, kind='hist', density=True, alpha=0.3, color='red', bins=30)
                min_gen_dists_df.plot(ax=ax, kind='kde', color='purple', bw_method=0.3)
                ax.legend(labels=['Real', 'Real KDE', 'Synthetic', 'Synthetic KDE'], loc='upper right', fontsize=36)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 5)
                ax.set_xlabel('Distances', fontsize=36)
                ax.set_ylabel('Density', fontsize=36)
                plt.grid(True)
                real_dataset_name = 'Diabetes_H' if dataset_name == '3' else 'Heart'
                title_iid = '_IID' if iid else '_Non-IID'
                ax.set_title('Distances (' + real_dataset_name + title_iid +')', fontsize=40)
                plt.yticks(fontsize=28)
                plt.xticks(fontsize=28)
                plt.tight_layout()
                plt.savefig(os.path.join(results_path, dir_name) + os.sep + exp_type + '_distances_'+ real_dataset_name + title_iid +'.pdf')
                plt.show()
                plt.close()

                # Should never be cero because it would mean that the real sample is the same as the synthetic one
                print(dir_name)
                print('Wilcoxon test between real and synthetic data: ', stats.wilcoxon(min_real_dists - min_gen_dists, alternative='less').pvalue)
                print('KS test between real and synthetic data: ', stats.kstest(min_real_dists, min_gen_dists, alternative='greater').pvalue)

    # STEPS (CONCERN 3)
    if concern_3:
        # Data distributions entre nodos)
        data_per_node = (100, 1000, 1000)
        m = 7500
        l = 1000
        round = 4
        node_name = '0'
        dataset_names = ['3', '7']
        n_nodes = len(data_per_node)
        n_samples_val = m + 2 * l

        # AUXILIAR INFORMATION
        # Train random forest with real data and get feature importance
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        for iid in [True, False]:
            for dataset_name in dataset_names:
                if iid:
                    dir_name = dataset_name + '_iid'
                else:
                    dir_name = dataset_name + '_niid'

                # Load real data
                real_df = pd.read_csv(os.path.join(results_path, dir_name, 'val_data_' + str(node_name) + '.csv'))
                y_real = real_df.iloc[:, -1].values
                x_real = real_df.iloc[:, :-1].values

                # Split data
                x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(x_real, y_real,
                                                                                        test_size=0.2,
                                                                                        random_state=0)

                # Train random forest
                model = RandomForestClassifier(n_estimators=100)
                model.fit(x_train_real, y_train_real)

                # Get accuracy
                acc_real_real = accuracy_score(y_test_real, model.predict(x_test_real))
                print('Accuracy real data: ', acc_real_real)

                # Get feature importance
                feature_importance = model.feature_importances_
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sort = np.argsort(feature_importance).tolist()
                print('Feature importance: ', feature_importance)


                # 1. Load real and synthetic data
                alone_syn_df = pd.read_csv(os.path.join(results_path, dir_name, 'public_data_alone_r_' + str(0) + '_n_' + node_name + '.csv'))
                favg_syn_df = pd.read_csv(os.path.join(results_path, dir_name, 'public_data_favg_r_' + str(round) + '_n_' + node_name + '.csv'))
                drs_syn_df = pd.read_csv(os.path.join(results_path, dir_name, 'public_data_drs_r_' + str(round) + '_n_' + node_name + '.csv'))
                real_df = pd.read_csv(os.path.join(results_path, dir_name, 'val_data_' + node_name + '.csv'))
                df_list = [real_df, alone_syn_df, favg_syn_df, drs_syn_df]
                len_data = alone_syn_df.shape[0]

                original_data = pd.read_csv('./data/3_data_raw.csv')

                # 2. Histograms for each node data
                # For each column, plot the histogram of the real data and the synthetic data
                for idx in sort[-6:]:
                    # get col name by index
                    col = real_df.columns[idx]

                    # Check if column is categorical and plot histogram instead of KDE
                    fig, ax = plt.subplots(figsize=(10, 8))

                    if original_data.loc[:, col].unique().all() == original_data.loc[:, col].unique().astype(int).all() and len(original_data.loc[:, col].unique()) < 10:

                        # Transform vae data
                        real_cats = real_df.loc[:, col].unique()
                        real_df_col = real_df.loc[:, col]
                        favg_syn_df_col = favg_syn_df.loc[:, col]
                        drs_syn_df_col = drs_syn_df.loc[:, col]
                        for idx, _ in enumerate(real_df.loc[:, col]):
                            # Take value in favg and drs and get closest value in real data
                            favg_val = favg_syn_df.loc[idx, col]
                            drs_val = drs_syn_df.loc[idx, col]

                            # Get closest value in real data
                            favg_syn_df_col[idx] = real_cats[np.argmin(np.abs(real_cats - favg_val))]
                            drs_syn_df_col[idx] = real_cats[np.argmin(np.abs(real_cats - drs_val))]

                        ax.hist([real_df_col, favg_syn_df_col, drs_syn_df_col], alpha=1, bins=20,  histtype='bar', stacked=False, color=['skyblue', 'orange', 'purple'], label=['Real', 'FedAvg', 'DRS'])
                    else:
                        sns.kdeplot(real_df.loc[:, col], fill=True, linewidths=1, color='skyblue', ax=ax)
                        sns.kdeplot(favg_syn_df.loc[:, col],  color='orange', ax=ax)
                        sns.kdeplot(drs_syn_df.loc[:, col], color='purple', ax=ax)

                    title_iid = '_IID' if iid else '_Non-IID'
                    d_name = 'Diabetes_H' + title_iid if dataset_name == '3' else 'Heart' + title_iid
                    plt.title(col + ' for ' + d_name, fontsize=40)
                    if col == 'Age':
                        ax.legend(['Real', 'FedAvg', 'DRS'], loc='lower left', fontsize=32)
                    else:
                        ax.legend(['Real', 'FedAvg', 'DRS'], loc='best', fontsize=32)
                    ax.set_xlabel(col.upper() + ' distribution', fontsize=32)
                    ax.set_ylabel('Frequency', fontsize=32)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_path, dir_name) + os.sep + 'hist_' + col + '_' + d_name + '.pdf')
                    plt.show()
                    plt.close()

    # STEPS (CONCERN 5)
    if concern_5:
        # Data distributions entre nodos )
        data_per_node = (100, 1000, 10000)
        m = 7500
        l = 1000
        n_nodes = len(data_per_node)
        n_samples_val = m + 2 * l
        for iid in [True, False]:
            for dataset_name in dataset_names:
                if iid:
                    dir_name = dataset_name + '_iid'
                else:
                    dir_name = dataset_name + '_niid'

                # 1. Load each node data. Should I just use val_data? It has the same distribution as train data but same number of samples for every node
                df_train_list, df_val_list, feat_distributions, clas, label = load_data(dataset_name, data_per_node,
                                                                                        iid, n_samples_val)

                # 2. Represent BMI columns for each node data
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.kdeplot(df_train_list[0].loc[:, 'BMI'], fill=True, linewidths=1, color='blue', ax=ax)
                sns.kdeplot(df_train_list[1].loc[:, 'BMI'], fill=True, linewidths=1, color='red', ax=ax)
                sns.kdeplot(df_train_list[2].loc[:, 'BMI'], fill=True, linewidths=1, color='purple', ax=ax)

                title_iid = '_IID' if iid else '_Non-IID'
                d_name = 'Diabetes_H' + title_iid if dataset_name == '3' else 'Heart' + title_iid
                ax.set_xlim(-4, 4)
                plt.title('BMI column for ' + d_name)
                ax.legend(['Node_0', 'Node_1', 'Node_2'], loc='upper right')
                ax.set_xlabel('BMI Distribution')
                plt.grid(True)
                plt.savefig(os.path.join(results_path, dir_name) + os.sep + 'kde_' + d_name + '.pdf')
                plt.show()
                plt.close()

