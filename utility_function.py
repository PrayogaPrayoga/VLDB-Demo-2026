from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import time

def SGD_class(X_train, Y_train, X_test, Y_test, iter=1000000, tolerance=1e-7, seed=None, scale = "min_max"):
    if seed is None:
        seed = np.random.randint(0, 100000)

    # Scale the data
    if scale == "min_max":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif scale == "standard":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
        X_train_scaled = X_train

    # Train SVM with SGD
    model = SGDClassifier(loss='hinge', max_iter=iter, tol=tolerance, fit_intercept=True, random_state=seed, warm_start=True)#RUN 2 fit intercept is true
    start = time.time()
    model.fit(X_train_scaled, Y_train)
    duration = time.time() - start
    # print(f"Training time: {duration:.2f} seconds") # Kept original comment

    train_accuracy = model.score(X_train_scaled, Y_train)
    test_accuracy = model.score(X_test_scaled, Y_test)

    return train_accuracy, test_accuracy, duration