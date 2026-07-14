import importlib.util
import re
import sys
import time

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import utility_function as ut  # local MinPrep version (MinMaxScaler)

# ── NEURIPS_2026 MR imports ───────────────────────────────────────────────────
_NEURIPS_PATH = '/data/prayoga/MI/SVM/synthetic/NEURIPS_2026/'
if _NEURIPS_PATH not in sys.path:
    sys.path.insert(0, _NEURIPS_PATH)

from Minimal_Impute import findminimalImputation

_ut_nr_spec = importlib.util.spec_from_file_location(
    'ut_nr', _NEURIPS_PATH + 'utility_function.py')
ut_nr = importlib.util.module_from_spec(_ut_nr_spec)
_ut_nr_spec.loader.exec_module(ut_nr)  # StandardScaler, no scale param


# ─── Data Utilities ───────────────────────────────────────────────────────────

def get_Xy(data, label):
    return data.drop(label, axis=1), data[label]


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    data_types = df.dtypes
    tbl = pd.concat([mis_val, mis_val_percent, data_types], axis=1)
    tbl = tbl.rename(columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Data Type'})
    tbl = tbl[tbl.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    return tbl


def get_single_value_columns(df):
    return df.columns[df.nunique() == 1].tolist()


def drop_label_with_null(df, column_name):
    return df.dropna(subset=[column_name])


# ─── Categorical Handling ──────────────────────────────────────────────────────

def manual_categorical_imputation(df, categorical_columns):
    df = df.reset_index(drop=True)
    df[categorical_columns] = df[categorical_columns].fillna("missing")
    encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_array = encoder.fit_transform(df[categorical_columns]).toarray()
    encoded_data = pd.DataFrame(one_hot_array, columns=encoder.get_feature_names_out(), index=df.index)
    feature_names = encoder.get_feature_names_out()
    missing_indicator_cols = [col for col in feature_names if '_missing' in col]
    for categorical_col in categorical_columns:
        missing_indicator_col = f"{categorical_col}_missing"
        if missing_indicator_col in missing_indicator_cols:
            mask = (encoded_data[missing_indicator_col] == 1)
            cols_to_replace = [col for col in encoded_data.columns if col.startswith(categorical_col)]
            encoded_data.loc[mask, cols_to_replace] = np.nan
            encoded_data.drop(columns=[missing_indicator_col], inplace=True)
    df.drop(columns=categorical_columns, inplace=True)
    return pd.concat([df, encoded_data], axis=1)


def drop_categorical_columns(df, conversion=False, featurize=False):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if conversion:
        for col in categorical_columns:
            most_common_value = df[col].mode().iloc[0]
            df[col] = df[col].apply(
                lambda x: 1 if pd.notna(x) and x == most_common_value else (0 if pd.notna(x) else x))
        return df.copy()
    elif featurize:
        for col in categorical_columns:
            if df[col].nunique() > 20:
                df.drop(columns=[col], inplace=True)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return manual_categorical_imputation(df, categorical_columns)
    else:
        return df.drop(categorical_columns, axis=1)


def preprocess_for_training(df, feature_columns, label):
    """Restrict df to the selected input features + target, one-hot encode
    any non-numeric feature columns (no feature is ever dropped), then split
    into train/test. Returns (X_train, Y_train, X_test, y_test)."""
    selected = df[list(feature_columns) + [label]].copy()
    features = selected.drop(columns=[label])
    target = selected[label]

    categorical_columns = features.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_columns:
        features = manual_categorical_imputation(features, categorical_columns)

    processed = pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)

    X_train, Y_train, X_test, y_test, _, _ = _prepare_split(processed, label)
    return X_train, Y_train, X_test, y_test


# ─── Certain Model ─────────────────────────────────────────────────────────────

def check_certain_model(X_train, y_train, X_test, y_test, verbose=False):
    res = True
    seed = np.random.randint(0, 100000)
    missing_data_indices = []
    missing_column_indices = []

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train

    missing_columns_indices = np.where(pd.DataFrame(X_train).isnull().any(axis=0))[0]
    missing_rows_indices = np.where(pd.DataFrame(X_train).isnull().any(axis=1))[0]

    X_train_missing_rows = X_train[missing_rows_indices]
    y_train_missing_rows = y_train[missing_rows_indices]

    X_train_complete = np.delete(X_train, missing_rows_indices, axis=0)
    y_train_complete = np.delete(y_train, missing_rows_indices, axis=0)

    scaler = MinMaxScaler()
    X_train_complete = scaler.fit_transform(X_train_complete)
    X_train_missing_rows = scaler.transform(X_train_missing_rows)
    X_test = scaler.transform(X_test)

    svm_model = SGDClassifier(
        loss="hinge", max_iter=1000000,
        fit_intercept=True, warm_start=True, random_state=seed
    )
    svm_model.fit(X_train_complete, y_train_complete)
    feature_weights = svm_model.coef_[0]

    for i in missing_columns_indices:
        if abs(feature_weights[i]) >= 1e-3:
            if verbose:
                missing_column_indices.append(i)
            else:
                res = False
                break

    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            if verbose:
                missing_data_indices.append(missing_rows_indices[i])
            else:
                res = False
                break

    if verbose and (len(missing_data_indices) > 0 or len(missing_column_indices) > 0):
        res = False

    cm_score = svm_model.score(X_test, y_test)
    missing_data_table = pd.DataFrame(X_train[missing_data_indices])
    print("cm: ", cm_score)

    return res, cm_score, missing_data_table, missing_data_indices


# ─── Baseline Imputers ─────────────────────────────────────────────────────────

def get_simple_imputer_model_classification(df_train, df_test, label):
    X_train, y_train = get_Xy(df_train, label)
    X_test, y_test = get_Xy(df_test, label)
    start = time.time()
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    meanimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    modeimputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for col in columns_with_nulls:
        if X_train[col].nunique() > 2:
            X_train[col] = meanimputer.fit_transform(X_train[[col]]).flatten()
        else:
            X_train[col] = modeimputer.fit_transform(X_train[[col]]).flatten()
    assert not X_train.isnull().any().any()
    _, score, _ = ut.SGD_class(X_train.values, y_train.values, X_test.values, y_test.values)
    return score, time.time() - start


def get_knn_imputer_model_classification(df_train, df_test, label):
    X_train, y_train = get_Xy(df_train, label)
    X_test, y_test = get_Xy(df_test, label)
    start = time.time()
    imputer = KNNImputer(missing_values=np.nan)
    imputed_X = imputer.fit_transform(X_train)
    assert not pd.DataFrame(imputed_X).isnull().any().any()
    _, score, _ = ut.SGD_class(imputed_X, y_train.values, X_test.values, y_test.values)
    return score, time.time() - start


def get_naive_imputer_model_classification(df_train, df_test, label):
    X_train, y_train = get_Xy(df_train, label)
    X_test, y_test = get_Xy(df_test, label)
    X_train_copy = X_train.copy()
    X_train_copy.dropna(inplace=True)
    y_train_aligned = y_train.loc[X_train_copy.index]
    assert not X_train_copy.isnull().any().any()
    start = time.time()
    _, score, _ = ut.SGD_class(X_train_copy.values, y_train_aligned.values, X_test.values, y_test.values)
    return score, time.time() - start


# ─── Main Pipelines ────────────────────────────────────────────────────────────

def _prepare_split(df, label):
    X, y = get_Xy(df, label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    df_train = pd.concat([X_train, Y_train], axis=1)
    df_test = pd.concat([X_test, Y_test], axis=1)
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    return X_train, Y_train, X_test, y_test, df_train, df_test


def certain_clean_main(X_train, Y_train, X_test, y_test, verbose=False):
    total_examples = len(X_train)
    rows_with_missing = len(X_train[X_train.isnull().any(axis=1)])
    missing_factor = rows_with_missing / total_examples

    start = time.time()
    result, CM_score, missing_data_table, missing_indices = check_certain_model(
        X_train.values, Y_train.values, X_test.values, y_test.values, verbose=verbose)
    CM_time = time.time() - start

    results = [
        {'Metric': 'Number of Rows with missing values', 'Value': rows_with_missing},
        {'Metric': 'Missing Factor',                     'Value': missing_factor},
        {'Metric': 'CM Result',                          'Value': 'Exists' if result else 'Does not Exist'},
        {'Metric': 'Running Time (CM)',                  'Value': CM_time},
        {'Metric': 'Accuracy (CM)',                      'Value': CM_score},
    ]
    return pd.DataFrame(results), missing_data_table, result, missing_indices


def MR_main(X_train, Y_train, X_test, y_test, batch_size=50, top_k=0.3, seed=42, verbose=True):
    total_examples = len(X_train)
    rows_with_missing = len(X_train[X_train.isnull().any(axis=1)])
    missing_factor = rows_with_missing / total_examples

    # Minimal Repair: use NEURIPS findminimalImputation to find which rows truly need repair
    mr_start = time.time()
    X_arr = X_train.reset_index(drop=True).values.astype(float)
    Y_arr = Y_train.reset_index(drop=True).values.copy()

    minimal_indices, _ = findminimalImputation(X_arr, Y_arr, batch_size, top_k, seed=seed)

    # Impute only the identified minimal rows (mean imputation)
    X_repaired = X_arr.copy()
    if len(minimal_indices) > 0:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(X_repaired)
        X_repaired[minimal_indices] = imputer.transform(X_repaired)[minimal_indices]

    # Drop any rows still containing NaNs (non-imputed missing rows)
    complete_mask = ~np.isnan(X_repaired).any(axis=1)
    X_final = X_repaired[complete_mask]
    Y_final = Y_arr[complete_mask]

    mr_scores = sorted(
        (ut_nr.SGD_class(X_final, Y_final, X_test.values, y_test.values)[1] for _ in range(10)),
        reverse=True,
    )
    mr_score = sum(mr_scores[:8]) / 8
    mr_time = time.time() - mr_start

    rows_repaired = len(minimal_indices)
    pct_repaired = rows_repaired / total_examples * 100
    missing_data_table = pd.DataFrame(X_arr[minimal_indices])

    results = [
        {'Metric': 'Number of Rows with missing values', 'Value': rows_with_missing},
        {'Metric': 'Missing Factor',                     'Value': missing_factor},
        # CM check is no longer recomputed here — certain_clean_main already determined
        # CM does not exist before MR_main is called, so re-running it was redundant.
        # {'Metric': 'CM Result',                          'Value': 'Exists' if result else 'Does not Exist'},
        # {'Metric': 'Running Time (CM)',                  'Value': CM_time},
        # {'Metric': 'Accuracy (CM)',                      'Value': CM_score},
        {'Metric': 'Rows Repaired (MR)',                 'Value': rows_repaired},
        {'Metric': '% Repaired (MR)',                    'Value': round(pct_repaired, 2)},
        {'Metric': 'Running Time (MR)',                  'Value': mr_time},
        {'Metric': 'Accuracy (MR)',                      'Value': mr_score},
    ]
    return pd.DataFrame(results), missing_data_table
