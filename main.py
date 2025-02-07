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
               n_samples_public, m, l, dataset_name, max_samples_train, exp_types, n_jobs=20, results_path='results',
               train=True):
    n_nodes = len(data_per_node)
    df_train_list, df_val_list, feat_distributions, clas, label = load_data(dataset_name, data_per_node, iid,
                                                                            n_samples_val)
    if iid:
        dir_name = dataset_name + '_iid'
    else:
        dir_name = dataset_name + '_niid'

    os.makedirs(os.path.join(results_path, dir_name), exist_ok=True)

    for exp_type in exp_types:
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

        if train:
            for round in range(nr):
                for node in nodes:
                    print(f"Round {round}/{nr}, Node {node.node_name}/{n_nodes}")
                    use_shared_data = round > 0  # Shared data is available only after the first round
                    tsne_path = results_path + os.sep + dir_name + os.sep + 'round_' + str(round) + '_node_' + str(
                        node.node_name) + '_real_gen_shared_samples.png'
                    training_info = node.train_on_local_data(n_epochs=epr, batch_size=128, lr=1e-3,
                                                             use_shared_data=use_shared_data,
                                                             max_samples_train=max_samples_train)
                    tr_info_per_node[int(node.node_name)].append(training_info)

                    # Generate data in the best node every round and in all nodes in the last round
                    if round == nr - 1:
                        _ = node.generate_public_data(
                            n_samples=n_samples_public)  # This data is used for validation and for federation in case of DRS
                        # Save the public df for later validation (i.e., the data generated)
                        node.public_df[0: m + 2 * l].to_csv(os.path.join(results_path, dir_name, 'public_data_' + exp_type + '_r_' + str(round) + '_n_' + node.node_name + '.csv'), index=False)

                    if 'drs' in exp_type and node.node_name == '2':
                        print('Generating data in best node to share them with the other nodes')
                        n = n_samples_public
                        _ = node.generate_public_data(n_samples=n)
                        node.public_df[0: m + 2 * l].to_csv(os.path.join(results_path, dir_name, 'public_data_' + exp_type + '_r_' + str(round) + '_n_' + node.node_name + '.csv'), index=False)

                if exp_type == 'drs':
                    # After all nodes have trained, share the best node's public data to the other two nodes before the next round
                    # shared_datas = [node.public_df for node in nodes]
                    shared_df = nodes[2].public_df
                    for i, node in enumerate(nodes):
                        if i != 2:
                            node.add_shared_data(shared_df)
                            # Save final shared df
                            node.shared_df.to_csv(os.path.join(results_path, dir_name,
                                                               'shared_data_' + exp_type + '_r_' + str(
                                                                   round) + '_n_' + node.node_name + '.csv'),
                                                  index=False)

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
                        os.path.join(results_path, dir_name,
                                     'model_' + exp_type + '_n_' + str(i) + '_m_' + str(j)) + '.pt')
            with open(os.path.join(results_path, dir_name, 'training_' + exp_type + '.pkl'), 'wb') as f:
                pickle.dump({'tr_info_per_node': tr_info_per_node}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Evaluation
        node_names = [str(i) for i in range(n_nodes)]
        round = 0 if exp_type == 'alone' else n_rounds - 1
        _ = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_node)(dataset_name, dir_name, exp_type, node_name, round, m, l, results_path=results_path,
                                   clas=clas, label=label, seed=seed) for node_name in node_names for seed in
            range(n_seeds_val))


def plot_experiment(dir_name, n_rounds, n_seeds, results_path, exp_types, n_nodes=3):
    # Load evaluation metrics for each node
    eval_data = {}
    nodes_real_real_mean = []
    nodes_real_real_std = []
    for n in range(n_nodes):
        eval_data[n] = {}
        for i, exp_type in enumerate(exp_types):
            eval_data[n][exp_type] = {}
            round = 0 if exp_type == 'alone' else n_rounds - 1
            eval_data[n][exp_type] = {}
            for seed in range(n_seeds):
                eval_data[n][exp_type][seed] = None
                results_path_file = os.path.join(results_path, dir_name,
                                                 'evaluation_' + exp_type + '_r_' + str(round) + '_n_' + str(
                                                     n) + '_s_' + str(seed) + '.csv')
                eval_data[n][exp_type][seed] = pd.read_csv(results_path_file)

        # We need to obtain the isolated, favg, and drs accuracies and compute the mean and std. There will only be one value pero node of acc real real
        nodes_real_real_mean.extend(
            [np.mean([eval_data[n][exp_type][seed]['acc_real_real'] for seed in range(n_seeds_val)]).item() for exp_type
             in exp_types])
        nodes_real_real_std.extend(
            [np.std([eval_data[n][exp_type][seed]['acc_real_real'] for seed in range(n_seeds_val)]).item() for exp_type
             in exp_types])

    # Present data
    table = []
    js_table = []
    for n in range(n_nodes):
        # Add empty row for good visualization
        table.append(['' for _ in range(7)])
        for i, exp_type in enumerate(exp_types):
            setting_row = ['' if i != 0 else 'Node ' + str(n + 1), exp_type]

            # Obtain divergence
            mean_js = np.mean([eval_data[n][exp_type][seed]['js'] for seed in range(n_seeds)])
            std_js = np.std([eval_data[n][exp_type][seed]['js'] for seed in range(n_seeds)])
            setting_row.append(str(np.round(mean_js, 3)) + ' (' + str(np.round(std_js, 3)) + ')')
            if exp_type == 'alone':
                js_table.append([mean_js.item()])
            else:
                js_table[-1].extend([mean_js.item()])

            # Significant difference
            if exp_type == 'alone':
                mean_iso = mean_js
                desv_iso = std_js
                setting_row.append('')
            else:
                js_test_gr = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds, mean_js, std_js, n_seeds,
                                                  equal_var=False, alternative='greater')
                js_test_le = ttest_ind_from_stats(mean_iso, desv_iso, n_seeds, mean_js, std_js, n_seeds,
                                                  equal_var=False, alternative='less')
                if js_test_gr.pvalue < 0.01:
                    setting_row.append('*')
                elif js_test_le.pvalue < 0.01:
                    setting_row.append('-')
                else:
                    setting_row.append('')

            # Obtain clinical utility validation accuracy
            setting_row.append(
                str(np.round(nodes_real_real_mean[n], 3)) + ' (' + str(np.round(nodes_real_real_std[n], 3)) + ')')
            gen_real_mean = np.mean([eval_data[n][exp_type][seed]['acc_gen_real'] for seed in range(n_seeds)])
            gen_real_std = np.std([eval_data[n][exp_type][seed]['acc_real_real'] for seed in range(n_seeds)])
            setting_row.append(str(np.round(gen_real_mean, 3)) + ' (' + str(np.round(gen_real_std, 3)) + ')')

            # Significant test
            acc_test_gr = ttest_ind_from_stats(nodes_real_real_mean[n], nodes_real_real_std[n], n_seeds, gen_real_mean,
                                               gen_real_std, n_seeds, equal_var=False, alternative='greater')
            acc_test_le = ttest_ind_from_stats(nodes_real_real_mean[n], nodes_real_real_std[n], n_seeds, gen_real_mean,
                                               gen_real_std, n_seeds, equal_var=False, alternative='less')
            if acc_test_gr.pvalue < 0.01:
                setting_row.append('-')
            elif acc_test_le.pvalue < 0.01:
                setting_row.append('**')
            else:
                setting_row.append('*')

            # Add row to table
            table.append(setting_row)

    headers = ['Node', 'Method', 'JS Divergence', 'JS p-value', 'Acc_real_real', 'Acc_gen_real', 'Acc p-value']
    print(f"Results for all nodes in experiment {dir_name}")
    print(tabulate.tabulate(table, headers, tablefmt='grid'))
    print(tabulate.tabulate(table, headers, tablefmt='latex'))
    print('\n')

    # Calculate MRR
    if len(exp_types) > 1:
        mrr_js_table = []
        ranks = []
        for idx, row in enumerate(js_table):
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

        headers = ['Node']
        mrr_js_table.append(['TOTAL'])
        for i, exp_type in enumerate(exp_types):
            headers.append(exp_type)
            mrr_js_table[-1].extend([mrr_vals[i]])
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
    n_seeds_train = 1
    n_seeds_val = 3
    dataset_names = ['3', '7']
    n_rounds = 5
    n_epochs_per_round = 200
    max_samples_train = max(data_per_node)  # Maximum number of samples to use for training: set so that best node does not use data from other nodes!
    results_path = 'results'
    exp_types = ['alone', 'favg', 'drs']

    train_flag = not True
    for dataset_name in dataset_names:
        if train_flag:
            # Using iid data
            experiment(data_per_node=data_per_node, iid=True, n_samples_val=n_samples_val,
                       n_seeds_train=n_seeds_train,
                       n_seeds_val=n_seeds_val, n_rounds=n_rounds, n_epochs_per_round=n_epochs_per_round,
                       n_samples_public=n_samples_val, m=m, l=l, dataset_name=dataset_name,
                       max_samples_train=max_samples_train, results_path=results_path, exp_types=exp_types)

            # Using non-iid data
            experiment(data_per_node=data_per_node, iid=False, n_samples_val=n_samples_val, n_seeds_train=n_seeds_train,
                       n_seeds_val=n_seeds_val, n_rounds=n_rounds, n_epochs_per_round=n_epochs_per_round,
                       n_samples_public=n_samples_val, m=m, l=l, dataset_name=dataset_name,
                       max_samples_train=max_samples_train, results_path=results_path, exp_types=exp_types)

        # Do the plots
        exp_types = ['alone', 'favg', 'drs']
        plot_experiment(dataset_name + '_iid', n_rounds, n_seeds_val, results_path, exp_types)
        plot_experiment(dataset_name + '_niid', n_rounds, n_seeds_val, results_path, exp_types)

    print('done')
