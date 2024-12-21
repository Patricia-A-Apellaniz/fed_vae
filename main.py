# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/10/2024

# Importing libraries
import os
import pickle
import tabulate

import numpy as np
import pandas as pd

from data_utils import load_data
from scipy.stats import rankdata
from joblib import Parallel, delayed
from nodes import Node, evaluate_node
from scipy.stats import ttest_ind_from_stats


def experiment(data_per_node, iid, n_samples_val, n_seeds_train, n_seeds_val, n_rounds, n_epochs_per_round,
               n_samples_public, m, l, dataset_name, max_samples_train, n_jobs=10, results_path='results'):
    n_nodes = len(data_per_node)
    df_train_list, df_val_list, feat_distributions, clas, label = load_data(dataset_name, data_per_node, iid,
                                                                            n_samples_val)
    if iid:
        dir_name = dataset_name + '_iid'
    else:
        dir_name = dataset_name + '_niid'

    os.makedirs(os.path.join(results_path, dir_name), exist_ok=True)

    for exp_type in ['alone', 'favg', 'drs']:

        if exp_type == 'favg':
            nst = 1  # For fed_avg, we use a single training seed
        else:
            nst = n_seeds_train

        if exp_type == 'alone':
            nr = 1  # For alone, we use a single round, with all the computational budget
            epr = n_epochs_per_round * n_rounds
        else:
            nr = n_rounds
            epr = n_epochs_per_round

        nodes = []
        for i in range(n_nodes):
            nodes.append(Node(df_train_list[i], df_val_list[i], feat_distributions, nst, latent_dim=20, hidden_size=256,
                              dataset_name=dataset_name, node_name=str(i)))
            # Save the validation df for each node (we validate outside the train loop for speed). Note that the validation data is the same for all exp_types, so do not be afraid to overwrite it
            df_val_list[i][0: m + 2 * l].to_csv(
                os.path.join(results_path, dir_name, 'val_data_' + nodes[i].node_name + '.csv'), index=False)
        tr_info_per_node = [[] for _ in range(n_nodes)]

        print(f"Starting training of experiment {dir_name} with {n_nodes} nodes and {nr} rounds, exp_type={exp_type}")

        for round in range(nr):
            for node in nodes:
                print(f"Round {round}/{nr}, Node {node.node_name}/{n_nodes}")
                use_shared_data = round > 0  # Shared data is available only after the first round
                training_info = node.train_on_local_data(n_epochs=epr, batch_size=1024, lr=1e-3,
                                                         use_shared_data=use_shared_data,
                                                         max_samples_train=max_samples_train, max_bgm_points=None)
                tr_info_per_node[int(node.node_name)].append(training_info)
                _ = node.generate_public_data(
                    n_samples=n_samples_public)  # This data is used for validation and for federation in case of DRS
                # Save the public df for later validation (i.e., the data generated)
                node.public_df[0: m + 2 * l].to_csv(os.path.join(results_path, dir_name,
                                                                 'public_data_' + exp_type + '_r_' + str(
                                                                     round) + '_n_' + node.node_name + '.csv'),
                                                    index=False)
            if exp_type == 'drs':
                # After all nodes have trained, share the public data to all nodes before the next round
                shared_datas = [node.public_df for node in nodes]
                for i, node in enumerate(nodes):
                    node.add_shared_data(pd.concat([shared_datas[j] for j in range(n_nodes) if j != i], axis=0))

            elif exp_type == 'favg':  # Fed-Avg: update the weights on all models. Note that there is a single training seed in fed_avg
                weights = []
                ns = []
                for node in nodes:
                    weights.append(node.models[0].state_dict())
                    ns.append(len(node.private_df))
                # Compute the average weights, take into account the number of samples
                avg_weights = {}
                for key in weights[0].keys():
                    avg_weights[key] = sum([ns[i] * weights[i][key] for i in range(len(weights))]) / sum(ns)
                # Update the weights
                for node in nodes:
                    node.models[0].load_state_dict(avg_weights)

        # Save trained models
        for i, node in enumerate(nodes):
            for j, model in enumerate(node.models):
                model.save(
                    os.path.join(results_path, dir_name, 'model_' + exp_type + '_n_' + str(i) + '_m_' + str(j)) + '.pt')
        with open(os.path.join(results_path, dir_name, 'training_' + exp_type + '.pkl'), 'wb') as f:
            pickle.dump({'tr_info_per_node': tr_info_per_node}, f, protocol=pickle.HIGHEST_PROTOCOL)

        node_names = [str(i) for i in range(n_nodes)]
        _ = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_node)(dataset_name, dir_name, exp_type, node_name, round, m, l, clas=clas, label=label,
                                   seed=seed) for node_name in node_names for seed in range(n_seeds_val) for round in
            range(nr))


def plot_experiment(dir_name, n_nodes, n_rounds, n_seeds_val, results_path='results'):
    def get_data(exp_type):
        with open(os.path.join(results_path, dir_name, 'training_' + exp_type + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        if exp_type == 'alone':
            dfs = [[pd.read_csv(os.path.join(results_path, dir_name,
                                             'evaluation_' + exp_type + '_r_' + str(0) + '_n_' + str(i) + '_s_' + str(
                                                 seed) + '.csv')) for seed in range(n_seeds_val)] for i in
                   range(n_nodes)]
        else:
            dfs = [[[pd.read_csv(os.path.join(results_path, dir_name,
                                              'evaluation_' + exp_type + '_r_' + str(r) + '_n_' + str(i) + '_s_' + str(
                                                  seed) + '.csv')) for seed in range(n_seeds_val)] for r in
                    range(n_rounds)] for i in range(n_nodes)]
        return data, dfs

    data_alone, dfs_alone = get_data('alone')
    data_drs, dfs_drs = get_data('drs')
    data_favg, dfs_favg = get_data('favg')

    # Show the results in table format
    table = []
    js_vals = []
    for i in range(n_nodes):
        # Print one table per node comparing all methods
        # Alone case: compute average js and accuracies
        alone_row = ['', 'isolated']
        desv_iso = np.std([dfs_alone[i][seed]['js'] for seed in range(n_seeds_val)])
        mean_iso = np.mean([dfs_alone[i][seed]['js'] for seed in range(n_seeds_val)])
        alone_row.append(str(np.round(mean_iso, 3)) + ' (' + str(np.round(desv_iso, 3)) + ')')
        alone_row.append('')
        js_vals.append([mean_iso.item()])

        # Clinical utility validation acc real real
        # We need to obtain the isolated, favg and drs accuracies and compute the mean and std. There will only be one value pero node of acc real real
        acc_real_real_iso = np.mean([dfs_alone[i][seed]['acc_real_real'] for seed in range(n_seeds_val)])
        acc_real_real_favg = np.mean([dfs_favg[i][-1][seed]['acc_real_real'] for seed in range(n_seeds_val)])
        acc_real_real_drs = np.mean([dfs_drs[i][-1][seed]['acc_real_real'] for seed in range(n_seeds_val)])
        acc_real_real_mean = np.mean([acc_real_real_iso, acc_real_real_favg, acc_real_real_drs])
        acc_real_real_std = np.std([acc_real_real_iso, acc_real_real_favg, acc_real_real_drs])

        # Clinical Utility Validation
        alone_row.append('')
        acc_desv_iso_gen_real = np.std([dfs_alone[i][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        acc_mean_iso_gen_real = np.mean([dfs_alone[i][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        alone_row.append(str(np.round(acc_mean_iso_gen_real, 3)) + ' (' + str(np.round(acc_desv_iso_gen_real, 3)) + ')')
        alone_acc_test_gr = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                                 acc_mean_iso_gen_real, acc_desv_iso_gen_real, n_seeds_val,
                                                 equal_var=False, alternative='greater')
        alone_acc_test_le = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                                 acc_mean_iso_gen_real, acc_desv_iso_gen_real, n_seeds_val,
                                                 equal_var=False, alternative='less')
        if alone_acc_test_gr.pvalue < 0.01:
            alone_row.append('-')
        elif alone_acc_test_le.pvalue < 0.01:
            alone_row.append('**')
        else:
            alone_row.append('*')

        table.append(alone_row)

        # Fed-avg case: compute average js and accuracies
        favg_row = ['Node ' + str(i + 1), 'favg']
        desv_favg = np.std([dfs_favg[i][-1][seed]['js'] for seed in range(n_seeds_val)])
        mean_favg = np.mean([dfs_favg[i][-1][seed]['js'] for seed in range(n_seeds_val)])
        favg_row.append(str(np.round(mean_favg, 3)) + ' (' + str(np.round(desv_favg, 3)) + ')')
        fedavg_js_test_gr = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds_val, mean_favg, desv_favg, n_seeds_val,
                                                 equal_var=False, alternative='greater')
        fedavg_js_test_le = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds_val, mean_favg, desv_favg, n_seeds_val,
                                                 equal_var=False, alternative='less')
        if fedavg_js_test_gr.pvalue < 0.01:
            favg_row.append('*')
        elif fedavg_js_test_le.pvalue < 0.01:
            favg_row.append('-')
        else:
            favg_row.append('')
        js_vals[-1].extend([mean_favg.item()])

        # Clinical Utility Validation
        favg_row.append(str(np.round(acc_real_real_mean, 3)) + ' (' + str(np.round(acc_real_real_std, 3)) + ')')
        acc_desv_favg_gen_real = np.std([dfs_favg[i][-1][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        acc_mean_favg_gen_real = np.mean([dfs_favg[i][-1][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        favg_row.append(
            str(np.round(acc_mean_favg_gen_real, 3)) + ' (' + str(np.round(acc_desv_favg_gen_real, 3)) + ')')
        favg_acc_test_gr = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                                acc_mean_favg_gen_real,
                                                acc_desv_favg_gen_real, n_seeds_val, equal_var=False,
                                                alternative='greater')
        favg_acc_test_le = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                                acc_mean_favg_gen_real,
                                                acc_desv_favg_gen_real, n_seeds_val, equal_var=False,
                                                alternative='less')
        if favg_acc_test_gr.pvalue < 0.01:
            favg_row.append('-')
        elif favg_acc_test_le.pvalue < 0.01:
            favg_row.append('**')
        else:
            favg_row.append('*')

        table.append(favg_row)

        # DRS case: compute average js and accuracies
        drs_row = ['', 'drs']
        desv_drs = np.std([dfs_drs[i][-1][seed]['js'] for seed in range(n_seeds_val)])
        mean_drs = np.mean([dfs_drs[i][-1][seed]['js'] for seed in range(n_seeds_val)])
        drs_row.append(str(np.round(mean_drs, 3)) + ' (' + str(np.round(desv_drs, 3)) + ')')
        drs_js_test_gr = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds_val, mean_drs, desv_drs, n_seeds_val,
                                              equal_var=False, alternative='greater')
        drs_js_test_le = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds_val, mean_drs, desv_drs, n_seeds_val,
                                              equal_var=False, alternative='less')
        if drs_js_test_gr.pvalue < 0.01:
            drs_row.append('*')
        elif drs_js_test_le.pvalue < 0.01:
            drs_row.append('-')
        else:
            drs_row.append('')
        js_vals[-1].extend([mean_drs.item()])

        # Clinical Utility Validation
        drs_row.append('')
        acc_desv_drs_gen_real = np.std([dfs_drs[i][-1][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        acc_mean_drs_gen_real = np.mean([dfs_drs[i][-1][seed]['acc_gen_real'] for seed in range(n_seeds_val)])
        drs_row.append(str(np.round(acc_mean_drs_gen_real, 3)) + ' (' + str(np.round(acc_desv_drs_gen_real, 3)) + ')')
        drs_acc_test_gr = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                               acc_mean_drs_gen_real,
                                               acc_desv_drs_gen_real, n_seeds_val, equal_var=False,
                                               alternative='greater')
        drs_acc_test_le = ttest_ind_from_stats(acc_real_real_mean, acc_real_real_std, n_seeds_val,
                                               acc_mean_drs_gen_real,
                                               acc_desv_drs_gen_real, n_seeds_val, equal_var=False, alternative='less')
        if drs_acc_test_gr.pvalue < 0.01:
            drs_row.append('-')
        elif drs_acc_test_le.pvalue < 0.01:
            drs_row.append('**')
        else:
            drs_row.append('*')

        table.append(drs_row)

    headers = ['Node', 'Method', 'JS', 'JS p-value', 'Acc_real_real', 'Acc_gen_real', 'Acc p-value']
    print(f"Results for all nodes in experiment {dir_name}")
    print(tabulate.tabulate(table, headers, tablefmt='grid'))
    print(tabulate.tabulate(table, headers, tablefmt='latex'))
    print('\n')

    # Calculate MRR
    mrr_js_table = []
    ranks = []
    for idx, row in enumerate(js_vals):
        ranking = rankdata(np.array([np.round(value, 2) for value in row]), method='dense')
        ranks.append(ranking)
        mrr_js_table.append(['Node ' + str(idx + 1)])
        mrr_js_table[-1].extend(ranking.tolist())

    # Second, calculate MRR for each case
    mrr_vals = []
    for model in range(len(ranks[0])):
        mrr_vals.append(0.0)
        for ranking in range(len(ranks)):
            mrr_vals[model] += 1 / (ranks[ranking][model])
        mrr_vals[model] /= len(ranks)

    mrr_js_table.append(['TOTAL', mrr_vals[0], mrr_vals[1], mrr_vals[2]])
    headers = ['Node', 'Isolated', 'FedAvg', 'DRS']
    print(f"JS Ranking for all nodes in experiment {dir_name}")
    print(tabulate.tabulate(mrr_js_table, headers, tablefmt='grid'))
    print(tabulate.tabulate(mrr_js_table, headers, tablefmt='latex'))
    print('\n')


if __name__ == '__main__':
    data_per_node = (100, 1000, 10000)  # Careful with changing this, as the non-iid case would change
    n_nodes = len(data_per_node)
    m = 7500
    l = 1000
    n_samples_val = m + 2 * l
    n_seeds_train = 3
    n_seeds_val = 3
    dataset_names = ['3', '7']
    n_rounds = 5
    n_epochs_per_round = 200
    max_samples_train = max(
        data_per_node)  # Maximum number of samples to use for training: set so that best node does not use data from other nodes!
    results_path = 'results_PAPER'

    train_flag = not True
    for dataset_name in dataset_names:
        if train_flag:
            # Using iid data
            experiment(data_per_node=data_per_node, iid=True, n_samples_val=n_samples_val, n_seeds_train=n_seeds_train,
                       n_seeds_val=n_seeds_val, n_rounds=n_rounds, n_epochs_per_round=n_epochs_per_round,
                       n_samples_public=n_samples_val, m=m, l=l, dataset_name=dataset_name,
                       max_samples_train=max_samples_train, results_path=results_path)

            # Using non-iid data
            experiment(data_per_node=data_per_node, iid=False, n_samples_val=n_samples_val, n_seeds_train=n_seeds_train,
                       n_seeds_val=n_seeds_val, n_rounds=n_rounds, n_epochs_per_round=n_epochs_per_round,
                       n_samples_public=n_samples_val, m=m, l=l, dataset_name=dataset_name,
                       max_samples_train=max_samples_train, results_path=results_path)

        # Do the plots
        plot_experiment(dataset_name + '_iid', n_nodes, n_rounds, n_seeds_val, results_path=results_path)
        plot_experiment(dataset_name + '_niid', n_nodes, n_rounds, n_seeds_val, results_path=results_path)

    print('done')
