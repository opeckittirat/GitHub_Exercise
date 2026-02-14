import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def train_closed_form(X, y):
    X = np.c_[np.ones(X.shape[0]), X]      # intercept
    theta = np.linalg.pinv(X) @ y          # stable closed-form
    return theta

def predict_closed_form(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]      # intercept
    return X @ theta

def poly_features(x, p):
    return np.vstack([x**k for k in range(1, p+1)]).T


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ['Unnamed: 0', 'zipcode', 'id', 'date']
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

x_train = train["sqft_living"].values.astype(float)
y_train = train["price"].values.astype(float)

x_test = test["sqft_living"].values.astype(float)
y_test = test["price"].values.astype(float)


results = []
for p in [1, 2, 3, 4, 5]:
    Xtr = poly_features(x_train, p)
    Xte = poly_features(x_test, p)

    theta = train_closed_form(Xtr, y_train)

    ytr_pred = predict_closed_form(Xtr, theta)
    yte_pred = predict_closed_form(Xte, theta)

    results.append([
        p,
        mean_squared_error(y_train, ytr_pred),
        r2_score(y_train, ytr_pred),
        mean_squared_error(y_test, yte_pred),
        r2_score(y_test, yte_pred),
    ])

# Print
for row in results:
    print(row)
