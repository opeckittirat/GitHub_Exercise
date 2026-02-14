import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(X, y, alpha=0.1, num_iters=100, theta_init=None):
    n, d = X.shape
    theta = np.zeros(d) if theta_init is None else theta_init.copy()

    for _ in range(num_iters):
        preds = X @ theta
        grad = (2/n) * (X.T @ (preds - y))
        theta = theta - alpha * grad

    return theta

def predict_linear(X, theta):
    return X @ theta

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ['Unnamed: 0', 'zipcode', 'id', 'date']
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

X_train_raw = train.drop(columns=["price"]).astype(float).values
y_train = train["price"].astype(float).values

X_test_raw = test.drop(columns=["price"]).astype(float).values
y_test = test["price"].astype(float).values

# Standardize features
mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0)
sigma[sigma == 0] = 1.0

X_train = (X_train_raw - mu) / sigma
X_test  = (X_test_raw  - mu) / sigma

# Add intercept
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test  = np.c_[np.ones(X_test.shape[0]),  X_test]

####################################
# 3 Experiment
alphas = [0.01, 0.1, 0.5]
iters_list = [10, 50, 100]

rows = []
for a in alphas:
    for iters in iters_list:
        theta = gradient_descent(X_train, y_train, alpha=a, num_iters=iters)

        ytr_pred = predict_linear(X_train, theta)
        yte_pred = predict_linear(X_test, theta)

        rows.append({
            "alpha": a,
            "iterations": iters,
            "train_mse": mean_squared_error(y_train, ytr_pred),
            "train_r2": r2_score(y_train, ytr_pred),
            "test_mse": mean_squared_error(y_test, yte_pred),
            "test_r2": r2_score(y_test, yte_pred),
            "theta": theta
        })

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

results_df = pd.DataFrame(rows)
print(results_df[["alpha","iterations","train_mse","train_r2","test_mse","test_r2"]])

