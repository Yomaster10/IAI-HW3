import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import RandomState
from sklearn.model_selection import KFold

"""
========================================================================
========================================================================
                Utility and Auxiliary functions
========================================================================
========================================================================
"""

ID = 123456789  # TODO: change it to your personal ID
random_gen = RandomState(seed=ID)
print_formatted_values = False


def set_formatted_values(value=True):
    global print_formatted_values
    print_formatted_values = value


def accuracy(y: np.array, y_pred: np.array):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    assert y.shape == y_pred.shape
    assert y.ndim == 1

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================

    return accuracy_val


def l2_dist(x1: np.array, x2: np.array):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a matrix of shape (N1, D).
    :param x2: Second samples matrix, a matrix of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """
    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Note: Use only basic numpy operations, no external code.

    dists = None

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================

    return dists


def load_data_set(clf_type: str):
    """
    Uses pandas to load train and test dataset.
    :param clf_type: a string equals 'ID3' or 'KNN'
    :return: A tuple of attributes_names (the features row) with train and test datasets split.
    """
    assert clf_type in ('ID3', 'KNN'), 'The parameter clf_type must be ID3 or KNN'
    hw_path = str(pathlib.Path(__file__).parent.absolute())
    dataset_path = hw_path + f"\\{clf_type}-dataset\\"
    train_file_path = dataset_path + "\\train.csv"
    test_file_path = dataset_path + "\\test.csv"
    # Import all columns omitting the fist which consists the names of the attributes
    train_dataset = pd.read_csv(train_file_path)
    test_dataset = pd.read_csv(test_file_path)
    attributes_names = list(pd.read_csv(train_file_path, delimiter=",", dtype=str, nrows=1).keys())
    return attributes_names, train_dataset, test_dataset


def create_train_validation_split(dataset: np.array, kf: KFold, validation_ratio: float = 0.2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.

    ----------------
    usage:

    ds_train, ds_valid = next(folds)
    x_train, y_train, x_valid, y_valid = get_dataset_split(ds_train, ds_valid, target_attribute)
    ----------------

    :param kf: KFold instance
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :return: A tuple of train and validation datasets split, as generator.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    #  Create two data loader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.

    for train_index, val_idx in kf.split(dataset):
        dl_train = dataset.loc[train_index]
        dl_valid = dataset.loc[val_idx]
        yield dl_train, dl_valid
    # ========================


def util_plot_graph(x, y, x_label, y_label, num_folds=5):
    """
    Auxiliary function for k-fold Cross-validation experiments
    """
    _, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(xticks=x))
    x_label = f'{x_label}'
    y_label = f'{y_label}'
    title_font = {'family': 'serif',
                  # 'color': 'darkred',
                  'weight': 'normal',
                  'size': 16,
                  }
    axis_font = {'family': 'serif',
                 'weight': 'normal',
                 'size': 14,
                 }

    plt.plot(x, y, 'r-d')  # plotting t, a separately
    ax.set_title(f'{num_folds}-fold Cross-validation', fontdict=title_font)
    plt.xlabel(x_label, fontdict=axis_font)
    plt.ylabel(y_label, fontdict=axis_font)
    plt.grid(True)
    plt.show()


def get_dataset_split(train_set: np.array, test_set: np.array, target_attribute: str):
    """
    Splits a dataset into a sample and label set, returning a tuple for each.
    :param train_set: train set
    :param test_set: test set
    :param target_attribute: attribute for classifying, the label column
    :return: A tuple of train and test datasets split.
    """
    # separate target from predictors
    x_train = np.array(train_set.drop(target_attribute, axis=1).copy())
    y_train = np.array(train_set[target_attribute].copy())

    x_test = np.array(test_set.drop(target_attribute, axis=1).copy())
    y_test = np.array(test_set[target_attribute].copy())

    return x_train, y_train, x_test, y_test
