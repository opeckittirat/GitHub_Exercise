import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)

def ridge_gradient_descent(X, y, alpha, num_iters, lam):
    n, d = X.shape
    theta = np.zeros(d)

    for _ in range(num_iters):
        preds = X @ theta
        error = preds - y

        grad = (2/n) * (X.T @ error)

        ridge_grad = 2 * lam * theta
        ridge_grad[0] = 0.0

        theta = theta - alpha * (grad + ridge_grad)

        if not np.isfinite(theta).all():
            return theta, False

    return theta, True

# data
N = 1000
x = np.random.uniform(-2, 2, size=N)
eps = np.random.normal(0, np.sqrt(2), size=N)  # N(0,2)
y = 1 + 2*x + eps

# standardize x
x_z = (x - x.mean()) / x.std()

X_design = np.c_[np.ones(N), x_z]

# lambdas for testing
lambdas = [0, 1, 10, 100, 1000, 10000]
num_iters = 5000

rows = []
for lam in lambdas:
    alpha = min(0.05, 0.25 / (lam + 1))

    theta, ok = ridge_gradient_descent(X_design, y, alpha=alpha, num_iters=num_iters, lam=lam)

    y_pred = X_design @ theta

    if not ok or (not np.isfinite(y_pred).all()):
        rows.append({
            "lambda": lam, "alpha": alpha, "iters": num_iters,
            "intercept": theta[0], "slope_(std_x)": theta[1],
            "MSE": np.nan, "R2": np.nan
        })
        continue

    rows.append({
        "lambda": lam, "alpha": alpha, "iters": num_iters,
        "intercept": theta[0], "slope": theta[1],
        "MSE": mean_squared_error(y, y_pred),
        "R2": r2_score(y, y_pred)
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
