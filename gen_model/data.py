# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/09/2023


# Packages to import
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------
#                   DATA SPLITTING
# -----------------------------------------------------------
def split_data(data, mask):
    train_data, val_data, train_mask, val_mask = train_test_split(data, mask, test_size=0.2, random_state=0)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    train_mask.reset_index(drop=True, inplace=True)
    val_mask.reset_index(drop=True, inplace=True)
    return train_data, train_mask, val_data, val_mask
