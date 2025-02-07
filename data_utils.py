# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/10/2024

# Importing libraries
import numpy as np
import pandas as pd


def load_data(dataset_name, data_per_node=(100, 1000, 10000), iid=True, n_samples_val=100):
    if dataset_name == '3':
        data = pd.read_csv('./data/3_data.csv')
        clas = True
        label = 'Diabetes'
    elif dataset_name == '7':
        data = pd.read_csv('./data/7_data.csv')
        clas = True
        label = 'HeartDiseaseorAttack'
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    feat_distributions = []
    for i in range(data.shape[1]):
        values = data.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 20 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', np.max(no_nan_values) + 1))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))

    n_nodes = len(data_per_node)
    df_train = []
    df_val = []
    if not iid:
        # Use BMI in the non-IID scenario of datasets 3 and 7
        assert dataset_name in ['3', '7'], "BMI is only available for datasets 3 and 7"
        assert n_nodes <= 3, "Non-IID scenario is only supported for up to 3 nodes"

        # Obtain data['BMI'] median
        median = data['BMI'].median()

        # Make a list of dataframes for each value of the BMI: above median and below median
        df_list = [data[data['BMI'] < median], data[data['BMI'] >= median]]
        prop_1 = np.array([0.9, 0.1])  # Evenly distributed (first node)
        prop_2 = np.array([0.1, 0.9])  # Many in the second group (second node)
        prop_3 = np.array([0.5, 0.5])  # Many in the first group (third node)

        # Now, assign the patients to the nodes
        i1, i2, i3 = 0, 0, 0
        for i, pr in enumerate([prop_1, prop_2, prop_3]):
            n1, n2 = int(pr[0] * data_per_node[i]), int(pr[1] * data_per_node[i])
            df_train.append(pd.concat([df_list[0].iloc[i1:i1 + n1], df_list[1].iloc[i2:i2 + n2]], axis=0))
            if df_train[-1].shape[0] > data_per_node[i]:
                df_train[-1] = df_train[-1].sample(n=data_per_node[i], random_state=42).reset_index(drop=True)
            else:
                df_train[-1] = df_train[-1].sample(frac=1, random_state=42).reset_index(drop=True)
            i1, i2 = i1 + n1, i2 + n2

            n1_val, n2_val = int(pr[0] * n_samples_val), int(pr[1] * n_samples_val)
            df_val.append(pd.concat([df_list[0].iloc[i1:i1 + n1_val], df_list[1].iloc[i2:i2 + n2_val]], axis=0))
            if df_val[-1].shape[0] > n_samples_val:
                df_val[-1] = df_val[-1].sample(n=n_samples_val, random_state=42).reset_index(drop=True)
            else:
                df_val[-1] = df_val[-1].sample(frac=1, random_state=42).reset_index(drop=True)

        for j, dft in enumerate(df_train):
            print(
                f"Node {j}: {dft.shape[0]} patients, prop of BMI < median: {np.sum(dft['BMI'] < data['BMI'].median()) / dft.shape[0]}, "
                f"prop of BMI >= median: {np.sum(dft['BMI'] >= data['BMI'].median()) / dft.shape[0]}")
    else:
        index = 0
        for i in range(n_nodes):
            df_train.append(data[index:index + data_per_node[i]])
            index += data_per_node[i]
            df_val.append(data[index:index + n_samples_val])
            index += n_samples_val
    return df_train, df_val, feat_distributions, clas, label
