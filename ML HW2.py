import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Drop columns if they exist
drop_cols = ['Unnamed: 0', 'zipcode', 'id', 'date']

train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

X_train = train.drop(columns=['price'])
y_train = train['price']

X_test = test.drop(columns=['price'])
y_test = test['price']

# Align columns
X_test = X_test.reindex(columns=X_train.columns)

model = LinearRegression()
model.fit(X_train, y_train)

# Train metrics
y_train_pred = model.predict(X_train)
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Training R2:", r2_score(y_train, y_train_pred))

# Test metrics
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("Testing MSE:", test_mse)
print("Testing R2:", test_r2)
print("Testing RMSE:", test_rmse)
