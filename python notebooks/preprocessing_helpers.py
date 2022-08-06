#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from data_info import *
from datetime import datetime

LSTM_STEP = 1
LSTM_FUTURE_TARGET = 1
LSTM_HISTORY = 6


def extract_data(train_file_path, columns, categorical_columns=CATEGORICAL_COLUMNS, categories_desc=CATEGORIES,
                 interpolate=True):
    # Read csv file and return
    all_data = pd.read_csv(train_file_path, usecols=columns)
    if categorical_columns is not None:
        # map categorical to columns
        for feature_name in categorical_columns:
            mapping_dict = {categories_desc[feature_name][i]: categories_desc[feature_name][i] for i in
                            range(0, len(categories_desc[feature_name]))}
            all_data[feature_name] = all_data[feature_name].map(mapping_dict)

        # Change mapped categorical data to 0/1 columns
        all_data = pd.get_dummies(all_data, prefix='', prefix_sep='')

    # fix missing data
    if interpolate:
        all_data = all_data.interpolate(method='linear', limit_direction='forward')

    return all_data


def generate_multivariate_data(dataset, history_size=LSTM_HISTORY, target_size=LSTM_FUTURE_TARGET,
                               step=LSTM_STEP, target_index=-1, target=None, single_step=False,
                               train_frac=TRAIN_DATASET_FRAC):
    datasets = []

    if target is None:
        target = dataset[:, target_index]
        dataset = dataset[:, :target_index]

    dataset_size = len(dataset)
    train_to_idx = int(dataset_size * train_frac) if train_frac != 1.0 else dataset_size - target_size
    start_train_idx = history_size
    start_val_idx = train_to_idx + history_size
    end_idx = dataset_size - target_size

    indexes = [(start_train_idx, train_to_idx)]
    if train_frac != 1.0:
        indexes.append((start_val_idx, end_idx))

    for (start_idx, end_idx) in indexes:
        data = []
        labels = []
        for i in range(start_idx, end_idx):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size - 2])
            else:
                labels.append(target[i:i + target_size - 1])

        datasets.append((np.array(data), np.array(labels)))

    return datasets


def generate_lstm_data(path, cols=CSV_COLUMNS + [DATETIME_COLUMN], label_column=LABEL_COLUMN, y_column=-1,
                       norm_cols=cols_to_norm, history_size=LSTM_HISTORY, target_size=LSTM_FUTURE_TARGET,
                       step=LSTM_STEP, cities=CATEGORIES['city'], index_col=DATETIME_COLUMN, single_step=False,
                       train_frac=TRAIN_DATASET_FRAC, train_scale=None, scale_cols=[], prepend_with_file=None,
                       extra_columns=[], group_by_column=False):
    dataset = extract_data(path, cols, categorical_columns=None)
    if prepend_with_file is not None:
        pre_dataset = extract_data(prepend_with_file, cols, categorical_columns=None)

    datasets = []

    scale = None

    if label_column not in dataset.columns:
        dataset[label_column] = pd.Series(np.zeros(len(dataset[DATETIME_COLUMN])), index=dataset.index)

    for city_name in cities:
        city_data = dataset[dataset['city'] == city_name]
        if prepend_with_file is not None:
            city_data = pre_dataset[pre_dataset['city'] == city_name].iloc[-(history_size+1):].append(city_data, ignore_index=True)
        if train_scale is None:
            train_scale = city_data.copy()
        city_data.index = city_data[index_col]
        city_data, scale = preproc_data(city_data[norm_cols + scale_cols + extra_columns + [label_column]], norm_cols=norm_cols,
                                        scale_cols=scale_cols, train_scale=train_scale)
        datasets.append(city_data.values)

    datasets = list(map(lambda x: generate_multivariate_data(x, target_index=y_column, single_step=single_step,
                                                             history_size=history_size, target_size=target_size,
                                                             step=step, train_frac=train_frac), datasets))

    if group_by_column:
        datasets = group_data_by_columns(datasets, columns=norm_cols + scale_cols + extra_columns)
        return datasets, scale, norm_cols + scale_cols + extra_columns
    return datasets, scale


def group_data_by_columns(datasets, columns):
    """
    :param datasets: [CxNxSxF]
    :param columns: F
    :return: CxNxFxS
    """
    new_dataset = []
    for i in range(len(datasets)):
        datalist = []
        for row in range(len(datasets[i][0][0])):
            row_data = []
            for column_idx in range(len(columns)):
                col_data = []
                for series in range(len(datasets[i][0][0][row])):
                    col_data.append(datasets[i][0][0][row][series][column_idx])

                row_data.append(col_data)
            datalist.append(row_data)

        new_dataset.append((datalist, datasets[i][0][1]))

    return new_dataset


def preproc_data(data, norm_cols=cols_to_norm, scale_cols=cols_to_scale, train_scale=None):
    # Make a copy, not to modify original data
    new_data = data.copy()
    if train_scale is None:
        train_scale = data
    if norm_cols:
        # Normalize temp and percipation
        new_data[norm_cols] = StandardScaler().fit(train_scale[norm_cols]).transform(new_data[norm_cols])

    if scale_cols:
        # Scale year and week no but within (0,1)
        new_data[scale_cols] = MinMaxScaler(feature_range=(0, 1)).fit(train_scale[scale_cols]).transform(
            new_data[scale_cols])

    return new_data, train_scale


def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                                  vocab))


def create_features_columns(data: pd.DataFrame, categorical_columns=CATEGORICAL_COLUMNS,
                            numerical_columns=NUMERIC_COLUMNS):
    feature_columns = []

    for feature_name in categorical_columns:
        vocabulary = data[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

    for feature_name in numerical_columns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                                dtype=tf.float32))

    return feature_columns


def make_input_fn(X, y, n_epochs=1, shuffle=True):
    num_examples = len(y)

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(num_examples)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(num_examples)
        return dataset

    return input_fn


def split_data(data, train_frac=TRAIN_DATASET_FRAC, label_column=LABEL_COLUMN, filter_cols=None):
    train_data = data.sample(frac=train_frac, random_state=0)
    test_data = data.drop(train_data.index)

    train_y = train_data.pop(label_column) if label_column is not None else []
    test_y = test_data.pop(label_column) if label_column is not None else []

    if filter_cols is not None:
        train_data = train_data[filter_cols]
        test_data = test_data[filter_cols]

    return (train_data, train_y), (test_data, test_y)


def export_test_to_csv(predictions=None, path=test_file, prefix='test'):
    print(len(predictions))
    print('asas')

    org_test_data = pd.read_csv(path)
    org_test_data['total_cases'] = predictions
    org_test_data['total_cases'] = org_test_data['total_cases'].apply(lambda x: int(x) if x > 0 else 0)
    org_test_data[['city', 'year', 'weekofyear', 'total_cases']].to_csv(
        'C:/Users/tessy/Desktop/Notes/SEM II/ML1/Project/' + prefix + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv", index=False)

def k_fold_data(x, y, folds=10):
    kfold = KFold(n_splits=folds, shuffle=True)
    return kfold.split(x, y)

