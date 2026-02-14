import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def train_closed_form(X, y):
    X = np.c_[np.ones(X.shape[0]), X]  # add intercept
    theta = np.linalg.pinv(X) @ y      # pseudo-inverse
    return theta

def predict_closed_form(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # add intercept
    return X @ theta

def poly_features(x, p):
    return np.vstack([x**k for k in range(1, p+1)]).T

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ['Unnamed: 0', 'zipcode', 'id', 'date']

train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

X_train = train.drop(columns=['price']).values
y_train = train['price'].values

X_test = test.drop(columns=['price']).values
y_test = test['price'].values


theta = train_closed_form(X_train, y_train)

y_train_pred = predict_closed_form(X_train, theta)
y_test_pred = predict_closed_form(X_test, theta)

print("Closed Form Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Closed Form Training R2:", r2_score(y_train, y_train_pred))

print("Closed Form Testing MSE:", mean_squared_error(y_test, y_test_pred))
print("Closed Form Testing R2:", r2_score(y_test, y_test_pred))

