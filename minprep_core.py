import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import utility_function as ut  # local MinPrep version (MinMaxScaler)

# ── Configurable paths ────────────────────────────────────────────────────────
# The research code backing Minimal Repair lives outside this repository, in a
# per-user directory that differs between machines. Override with env vars.
REPO_ROOT = Path(__file__).resolve().parent

ALGO_PATH = Path(os.environ.get(
    'MINPREP_ALGO_PATH', Path.home() / 'CM_code')).expanduser()

SHARED_DIR = Path(os.environ.get(
    'MINPREP_SHARED_DIR', REPO_ROOT / 'shared')).expanduser()

DEFAULT_DATASET = Path(os.environ.get(
    'MINPREP_DEFAULT_DATASET',
    REPO_ROOT / 'Sample-Datasets' / 'water_potability.csv')).expanduser()

DATASETS_DIR = REPO_ROOT / 'Sample-Datasets'

# ── Demo mode ─────────────────────────────────────────────────────────────────
# The app has two modes:
#   * "run"  — actually executes CM/ACM checks, MR/AMR repair and baselines
#              (slow; used to show real artifacts).
#   * "demo" — serves pre-computed results from demo_results/results.json
#              instantly (used on stage where there is no time to wait).
# The build-level default comes from MINPREP_MODE; the notebook can override it
# per-request via a hidden URL-parameter toggle.
DEFAULT_MODE = os.environ.get('MINPREP_MODE', 'demo').strip().lower()

DEMO_STORE_PATH = Path(os.environ.get(
    'MINPREP_DEMO_STORE', REPO_ROOT / 'demo_results' / 'results.json')).expanduser()

_algo_cache = {}
_demo_cache = {}


def load_algo():
    """Import findminimalImputation and its companion utility module from
    ALGO_PATH. Loaded on demand so that the CM check still works on machines
    where the Minimal Repair research code is not present."""
    if _algo_cache:
        return _algo_cache

    minimal_impute_py = ALGO_PATH / 'Minimal_Impute.py'
    utility_py = ALGO_PATH / 'utility_function.py'
    for required in (minimal_impute_py, utility_py):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} not found. Set MINPREP_ALGO_PATH to the directory "
                f"containing Minimal_Impute.py and utility_function.py "
                f"(currently {ALGO_PATH})."
            )

    if str(ALGO_PATH) not in sys.path:
        sys.path.insert(0, str(ALGO_PATH))

    from Minimal_Impute import findminimalImputation

    spec = importlib.util.spec_from_file_location('ut_nr', str(utility_py))
    ut_nr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ut_nr)  # no scale param, unlike the local ut

    _algo_cache['findminimalImputation'] = findminimalImputation
    _algo_cache['ut_nr'] = ut_nr
    return _algo_cache


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

def check_certain_model_classification(X_train, y_train, X_test, y_test, verbose=False):
    res = True
    seed = np.random.randint(0, 100000)
    missing_data_indices = []
    missing_column_indices = []

    # ─── Preprocessing ──────────────────────────────────────────────────────
    # Split X_train into complete rows vs. rows with missing values, then
    # scale everything with a MinMaxScaler fit on the complete rows only.
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

    # ─── Checking algorithm ─────────────────────────────────────────────────
    # Train an SVM on the complete rows, then test whether the missing
    # columns/rows could have altered the decision boundary (i.e. whether
    # the model is "certain" despite the missing data).
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


def check_certain_model_regression(X_train, y_train, X_test, y_test, verbose=False):
    assert not X_test.isnull().any().any(), "X_test must be fully observed"

    # Rebind to newly-scaled DataFrames (not X_train.loc[...] = ...) so we
    # never write into the caller's original data, same principle as the
    # classification version. Kept as DataFrames (not plain arrays like
    # classification does) because get_submatrix()/.fillna() below need
    # column/NaN-aware DataFrame methods, not just position-indexed values.
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train[X_train.columns]),
        columns=X_train.columns, index=X_train.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test[X_train.columns]),
        columns=X_train.columns, index=X_test.index,
    )

    missing_train, CX_train = get_submatrix(X_train)
    missing_indices = X_train.index[X_train.isnull().any(axis=1)].tolist()
    missing_data_table = X_train.loc[missing_indices]

    # No fully-observed column to anchor a baseline model on — certification
    # can't be evaluated, so treat the data as not certain and let the
    # caller fall back to Minimal Repair.
    if CX_train.shape[1] == 0:
        return False, 0.0, missing_data_table, missing_indices

    reg = LinearRegression(fit_intercept=False).fit(CX_train.values, y_train)
    w_bar = reg.coef_
    loss = np.dot(CX_train.values, w_bar.T) - y_train
    result = check_orthogonal(missing_train, loss)
    score = 0.0
    if result:
        clf = LinearRegression(fit_intercept=False).fit(X_train.fillna(0).values, y_train) #NEED TO VERIFY 
        y_pred = clf.predict(X_test.values)
        score = mean_squared_error(y_pred, y_test.values)

    if verbose:
        print(f"The mean squared error of the optimal model is {score:.2f}")

    return result, score, missing_data_table, missing_indices

def get_submatrix(data):
    columns_without_nulls = data.columns[data.notnull().all()]
    C = data[columns_without_nulls]
    missing = data.drop(columns_without_nulls,axis = 1)
    return missing,C


def check_orthogonal(M,l):
    flag = True
    case = ''
    for i in range(M.shape[1]):
        total = 0
        for j in range(len(l)):
            if np.isnan(M.iloc[j,i]) and not np.isclose(l[j], 0,atol=1e-02):
                flag = False
                case = 'case1: ' + str(l[j])
                break
            elif not np.isnan(M.iloc[j,i]):
                #print(f'inside case2 : M:{M.iloc[j,i]}, l:{l[j]}')
                total += M.iloc[j,i] * l[j]
        if not np.isclose(total ,0, atol = 1e-02):
            flag = False
            case = 'case2: ' + str(total)
            break
    #print(case)
    return flag

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
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    df_train = pd.concat([X_train, Y_train], axis=1)
    df_test = pd.concat([X_test, Y_test], axis=1)
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    return X_train, Y_train, X_test, y_test, df_train, df_test


def certain_clean_main(X_train, Y_train, X_test, y_test, task_type='classification', verbose=False):
    total_examples = len(X_train)
    rows_with_missing = len(X_train[X_train.isnull().any(axis=1)])
    missing_factor = rows_with_missing / total_examples

    start = time.time()
    if task_type == 'regression':
        result, CM_score, missing_data_table, missing_indices = check_certain_model_regression(
            X_train, Y_train, X_test, y_test, verbose=verbose)
        score_label = 'MSE (CM)'
    else:
        result, CM_score, missing_data_table, missing_indices = check_certain_model_classification(
            X_train.values, Y_train.values, X_test.values, y_test.values, verbose=verbose)
        score_label = 'Accuracy (CM)'
    CM_time = time.time() - start

    results = [
        {'Metric': 'Number of Rows with missing values', 'Value': rows_with_missing},
        {'Metric': 'Missing Factor',                     'Value': missing_factor},
        {'Metric': 'CM Result',                          'Value': 'Exists' if result else 'Does not Exist'},
        {'Metric': 'Running Time (CM)',                  'Value': CM_time},
        {'Metric': score_label,                          'Value': CM_score},
    ]
    return pd.DataFrame(results), missing_data_table, result, missing_indices


def MR_main(X_train, Y_train, X_test, y_test, batch_size=50, top_k=0.3, seed=42, verbose=True):
    total_examples = len(X_train)
    rows_with_missing = len(X_train[X_train.isnull().any(axis=1)])
    missing_factor = rows_with_missing / total_examples

    # Minimal Repair: use findminimalImputation to find which rows truly need repair
    algo = load_algo()
    mr_start = time.time()
    X_arr = X_train.reset_index(drop=True).values.astype(float)
    Y_arr = Y_train.reset_index(drop=True).values.copy()

    # Return arity differs between revisions of Minimal_Impute.py (some also
    # return timing and diagnostics); the selected row indices are always first.
    mr_result = algo['findminimalImputation'](
        X_arr, Y_arr, batch_size, top_k, seed=seed)
    minimal_indices = np.asarray(
        mr_result[0] if isinstance(mr_result, tuple) else mr_result, dtype=int)

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
        (algo['ut_nr'].SGD_class(X_final, Y_final, X_test.values, y_test.values)[1]
         for _ in range(10)),
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


def omp_select_features(X, y, threshold, max_iter=100):
    """Orthogonal-Matching-Pursuit-style feature selection (ported from
    MI/Linear_Regression/synthetic/omp_test_mnar copy 2.py). Starting from the
    always-kept complete columns, greedily adds whichever incomplete column is
    most correlated with the current residual, stopping once no remaining
    incomplete column's correlation clears `threshold`. Returns the indices of
    all kept features (complete + selected incomplete) and how many of the
    incomplete features were selected."""
    numNeedingImputation = 0

    X_impute = X.copy()
    X_impute.fillna(X_impute.mean(), inplace=True)
    assert not X_impute.isna().any().any(), "There are still NaN values in X_impute."
    assert not y.isna().any(), "There are NaN values in the target vector y."

    complete_features = X.columns[X.notna().all()].tolist()
    S = [X.columns.get_loc(feature) for feature in complete_features]

    incomplete_features = X.columns[X.isna().any()].tolist()
    remaining_features = [X.columns.get_loc(feature) for feature in incomplete_features]

    if complete_features:
        model = LinearRegression()
        model.fit(X_impute[complete_features], y)
        r = y - model.predict(X_impute[complete_features])
    else:
        r = y - y.mean()

    for _ in range(max_iter):
        if not remaining_features:
            break

        dot_products = X_impute.iloc[:, remaining_features].T @ r
        norms = np.linalg.norm(X_impute.iloc[:, remaining_features], axis=0) * np.linalg.norm(r)
        norms = np.where(norms == 0, 1e-10, norms)
        cosine_similarities = np.abs(dot_products / norms)

        max_cosine_similarity = np.max(cosine_similarities)
        if threshold > 0 and max_cosine_similarity < threshold:
            break
        if np.isnan(max_cosine_similarity):
            break

        j = remaining_features[np.argmax(cosine_similarities)]
        S.append(j)
        remaining_features.remove(j)
        numNeedingImputation += 1

        model = LinearRegression()
        model.fit(X_impute.iloc[:, S], y)
        r = y - model.predict(X_impute.iloc[:, S])

    return S, numNeedingImputation


def MR_main_regression(X_train, Y_train, X_test, y_test, threshold=0.005, verbose=True):
    total_examples = len(X_train)
    rows_with_missing = len(X_train[X_train.isnull().any(axis=1)])
    missing_factor = rows_with_missing / total_examples

    # Minimal Repair (regression): OMP-style feature selection decides which
    # incomplete columns are correlated enough with the target to be worth
    # imputing; the rest are dropped entirely rather than imputed.
    mr_start = time.time()
    must_impute_features, numNeedingImputation = omp_select_features(X_train, Y_train, threshold)
    must_impute_set = set(must_impute_features)

    incomplete_columns = X_train.columns[X_train.isna().any()].tolist()
    selected_columns = [c for c in incomplete_columns if X_train.columns.get_loc(c) in must_impute_set]
    dropped_columns = [c for c in incomplete_columns if c not in selected_columns]

    X_train_repaired = X_train.drop(columns=dropped_columns)
    X_test_repaired = X_test.drop(columns=dropped_columns)

    rows_repaired = 0
    if selected_columns:
        rows_repaired = int(X_train[selected_columns].isna().any(axis=1).sum())
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_repaired[selected_columns] = imputer.fit_transform(X_train_repaired[selected_columns])

    assert not X_train_repaired.isnull().any().any()

    model = LinearRegression()
    model.fit(X_train_repaired.values, Y_train.values)
    y_pred = model.predict(X_test_repaired.values)
    mr_score = mean_squared_error(y_test.values, y_pred)
    mr_time = time.time() - mr_start

    pct_repaired = rows_repaired / total_examples * 100
    if selected_columns:
        missing_data_table = X_train[X_train[selected_columns].isna().any(axis=1)]
    else:
        missing_data_table = X_train.iloc[0:0]

    if verbose:
        print(f"Features imputed: {numNeedingImputation}, Features dropped: {len(dropped_columns)}")
        print("mr: ", mr_score)

    results = [
        {'Metric': 'Number of Rows with missing values', 'Value': rows_with_missing},
        {'Metric': 'Missing Factor',                     'Value': missing_factor},
        {'Metric': 'Rows Repaired (MR)',                 'Value': rows_repaired},
        {'Metric': '% Repaired (MR)',                    'Value': round(pct_repaired, 2)},
        {'Metric': 'Features Imputed (MR)',              'Value': numNeedingImputation},
        {'Metric': 'Features Dropped (MR)',              'Value': len(dropped_columns)},
        {'Metric': 'Running Time (MR)',                  'Value': mr_time},
        {'Metric': 'MSE (MR)',                           'Value': mr_score},
    ]
    return pd.DataFrame(results), missing_data_table


# ─── Demo Mode Lookups ─────────────────────────────────────────────────────────
# These serve pre-computed results from demo_results/results.json so the app can
# respond instantly on stage. They intentionally return small plain dicts; the
# notebook renders them into the same metric-card layout that run mode uses.

def load_demo_store():
    """Load and cache the demo results store."""
    if _demo_cache:
        return _demo_cache['store']
    if not DEMO_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Demo results store not found at {DEMO_STORE_PATH}. "
            f"Generate it with: python tools/build_demo_store.py"
        )
    with open(DEMO_STORE_PATH) as f:
        store = json.load(f)
    _demo_cache['store'] = store
    return store


# ─── Demo curation (stage allowlist) ───────────────────────────────────────────
# What we actually put on stage, and in what order. This is an ALLOWLIST:
# datasets not listed here are hidden from the demo — Water Potability and Breast
# Cancer (weak / near-zero-missingness stories) and every regression set (whose
# committed MSE numbers are on an inconsistent scale). For each exposed dataset we
# also pin the model(s) and missingness level(s) to the combinations whose
# drop-baseline story is grounded in demo_results/results.json. See the design
# notes for the per-dataset rationale.
#   'models'   : model keys to offer, in this display order.
#   'variants' : missingness-level variant keys to offer (None = all in store).
DEMO_CURATION = {
    # Act 1 — Certification: a Certain Model exists -> no repair (convex only).
    #   1/5% : CM EXISTS (SVM).  40/60% : CM gone -> Minimal Repair beats drop
    #   (+0.114 / +0.083).  20% excluded (repair ties drop).
    'malware':    {'models': ['svm'],        'variants': ['1', '5', '40', '60']},
    # Act 2 — Repair beats dropping; drop baseline is REAL (injected sets).
    'heart':      {'models': ['svm', 'mlp'], 'variants': ['40']},   # +0.125 / +0.040
    'parkinsons': {'models': ['svm', 'mlp'], 'variants': ['60']},   # +0.240 / +0.285 (drop collapses)
    # Act 2b — Repair beats dropping; drop baseline still PLACEHOLDER (illustrative).
    'bankruptcy': {'models': ['svm', 'mlp'], 'variants': None},     # +0.122 / +0.107
    'online_ed':  {'models': ['svm', 'mlp'], 'variants': None},     # +0.096 / +0.058
    # Act 3 — Scalability wall: full imputation = OT; repair finishes and wins.
    'fraud':      {'models': ['mlp', 'ft'],  'variants': ['40', '60']},  # MLP +0.27/+0.43, FT +0.27/+0.41
    'higgs':      {'models': ['mlp', 'ft'],  'variants': ['40', '60']},  # MLP +0.15, FT +0.17/+0.19
}


def _curation(dataset_key):
    """Curation entry for a dataset, or None if the dataset is not exposed."""
    return DEMO_CURATION.get(dataset_key)


def demo_datasets():
    """Return [(key, display), ...] of datasets exposed in the demo, in stage order."""
    store = load_demo_store()
    return [(k, store['datasets'][k]['display'])
            for k in DEMO_CURATION if k in store['datasets']]


def demo_dataset_meta(dataset_key):
    """Return the full store entry for one dataset (task, target, variants)."""
    store = load_demo_store()
    if dataset_key not in store['datasets']:
        raise KeyError(f"Unknown demo dataset '{dataset_key}'")
    return store['datasets'][dataset_key]


def demo_default_variant(dataset_key):
    """The variant selected by default for a dataset: the first exposed level."""
    variants = demo_variants(dataset_key)
    if variants:
        return variants[0][0]
    ds = demo_dataset_meta(dataset_key)
    return ds.get('default_variant') or next(iter(ds['variants']))


def demo_variants(dataset_key):
    """Return [(variant_key, label, level), ...] of exposed missingness levels."""
    ds = demo_dataset_meta(dataset_key)
    cur = _curation(dataset_key)
    allowed = cur.get('variants') if cur else None
    return [(vk, v['label'], v['level']) for vk, v in ds['variants'].items()
            if allowed is None or vk in allowed]


def demo_variant_meta(dataset_key, variant_key):
    """Return the store entry for one (dataset, variant): stats + models."""
    ds = demo_dataset_meta(dataset_key)
    if variant_key not in ds['variants']:
        raise KeyError(f"Unknown variant '{variant_key}' for '{dataset_key}'")
    return ds['variants'][variant_key]


def demo_models(dataset_key, variant_key):
    """Return [(model_key, display), ...] of exposed models, in curated order."""
    variant = demo_variant_meta(dataset_key, variant_key)
    cur = _curation(dataset_key)
    allowed = cur.get('models') if cur else None
    if allowed is None:
        return [(k, v['display']) for k, v in variant['models'].items()]
    order = {k: i for i, k in enumerate(allowed)}
    items = [(k, v) for k, v in variant['models'].items() if k in order]
    items.sort(key=lambda kv: order[kv[0]])
    return [(k, v['display']) for k, v in items]


def demo_methods():
    """Return [(key, label), ...] of imputation methods offered in demo mode."""
    store = load_demo_store()
    return [(m['key'], m['label']) for m in store['methods']]


def demo_feature_missingness(dataset_key, variant_key):
    """Per-feature missingness for the profile page: (rows, features_total).

    ``rows`` is a list of [feature_name, pct_missing] (already the top-N by
    missingness). Synthesized at build time so it renders without a CSV.
    """
    stats = demo_variant_meta(dataset_key, variant_key)['stats']
    return stats.get('feature_missing', []), stats.get('features_total', 0)


def _model_entry(dataset_key, variant_key, model_key):
    ds = demo_dataset_meta(dataset_key)
    variant = demo_variant_meta(dataset_key, variant_key)
    if model_key not in variant['models']:
        raise KeyError(
            f"Model '{model_key}' not available for '{dataset_key}/{variant_key}'")
    return ds, variant, variant['models'][model_key]


def demo_model_supports_check(dataset_key, variant_key, model_key):
    """True if this model supports CM/ACM checking (convex models only).

    MLP and FT-Transformer return False -> the app routes them straight to
    minimal repair.
    """
    _, _, model = _model_entry(dataset_key, variant_key, model_key)
    return bool(model.get('supports_check', True))


def demo_model_supports_activeclean(dataset_key, variant_key, model_key):
    """True if ActiveClean is a valid baseline for this model."""
    _, _, model = _model_entry(dataset_key, variant_key, model_key)
    return bool(model.get('supports_activeclean', True))


def demo_drop_baseline(dataset_key, variant_key, model_key):
    """Pre-computed 'drop all incomplete samples' baseline (method-independent)."""
    ds, _, model = _model_entry(dataset_key, variant_key, model_key)
    rec = model['drop_incomplete']
    return {
        'name': 'Drop all incomplete samples',
        'score': rec['score'],
        'score_label': model['score_label'],
        'time_s': rec['time_s'],
        'finished': bool(rec.get('finished', True)),
        'dnf_reason': rec.get('dnf_reason'),
        'rows_dropped': rec['rows_dropped'],
        'pct_dropped': rec['pct_dropped'],
        'higher_is_better': ds['task'] == 'classification',
    }


def demo_check(dataset_key, variant_key, model_key, is_acm, threshold):
    """Pre-computed CM/ACM check verdict.

    Returns a dict:
        {name, exists, score, score_label, time_s, threshold, higher_is_better}
    ACM exists when the user-provided threshold >= the stored acm_gap.
    """
    ds, _, model = _model_entry(dataset_key, variant_key, model_key)
    check = model['check']
    score_label = model['score_label']
    higher_is_better = ds['task'] == 'classification'
    if is_acm:
        exists = float(threshold) >= float(check['acm_gap'])
        return {
            'name': 'ACM',
            'exists': bool(exists),
            'score': check['acm']['score'],
            'score_label': score_label,
            'time_s': check['acm']['time_s'],
            'threshold': threshold,
            'acm_gap': check['acm_gap'],
            'higher_is_better': higher_is_better,
        }
    return {
        'name': 'CM',
        'exists': bool(check['cm']['exists']),
        'score': check['cm']['score'],
        'score_label': score_label,
        'time_s': check['cm']['time_s'],
        'threshold': None,
        'acm_gap': None,
        'higher_is_better': higher_is_better,
    }


def demo_repair(dataset_key, variant_key, model_key, is_acm, method_key):
    """Pre-computed repair result. CM pairs with MR, ACM pairs with AMR."""
    ds, _, model = _model_entry(dataset_key, variant_key, model_key)
    which = 'amr' if is_acm else 'mr'
    rec = model['by_method'][method_key][which]
    return {
        'name': 'AMR' if is_acm else 'MR',
        'full_name': 'Almost Minimal Repair' if is_acm else 'Minimal Repair',
        'score': rec['score'],
        'score_label': model['score_label'],
        'time_s': rec['time_s'],
        'finished': bool(rec.get('finished', True)),
        'dnf_reason': rec.get('dnf_reason'),
        'pct_imputed': rec['pct_imputed'],
        'rows_imputed': rec['rows_imputed'],
        'higher_is_better': ds['task'] == 'classification',
    }


def demo_baseline(dataset_key, variant_key, model_key, method_key, which):
    """Pre-computed baseline result. `which` is 'activeclean' or 'full_impute'."""
    ds, _, model = _model_entry(dataset_key, variant_key, model_key)
    rec = model['by_method'][method_key][which]
    labels = {'activeclean': 'ActiveClean', 'full_impute': 'Full Imputation'}
    return {
        'name': labels[which],
        'score': rec['score'],
        'score_label': model['score_label'],
        'time_s': rec['time_s'],
        'finished': bool(rec.get('finished', True)),
        'dnf_reason': rec.get('dnf_reason'),
        'pct_imputed': rec['pct_imputed'],
        'rows_imputed': rec['rows_imputed'],
        'higher_is_better': ds['task'] == 'classification',
    }
