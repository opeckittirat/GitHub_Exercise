import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('spambase.data', header=None)

# Last column = label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy

print("Accuracy:", accuracy)
print("Error:", error)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

coefficients = pd.Series(model.coef_[0])

# Top positive (spam indicators)
top_positive = coefficients.sort_values(ascending=False).head(10)

# Top negative (non-spam indicators)
top_negative = coefficients.sort_values().head(10)

print("Top Positive Features:\n", top_positive)
print("\nTop Negative Features:\n", top_negative)

thresholds = [0.25, 0.5, 0.75, 0.9]

for t in thresholds:
    y_thresh = (y_prob >= t).astype(int)

    acc = accuracy_score(y_test, y_thresh)
    prec = precision_score(y_test, y_thresh)
    rec = recall_score(y_test, y_thresh)

    print(f"\nThreshold: {t}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)

print("question 2")
print("question 2")
print("question 2")
print("question 2")
print("question 2")

# Standardize features for gradient descent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias column
X_train_gd = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_gd = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Convert y to numpy arrays
y_train_gd = y_train.to_numpy()
y_test_gd = y_test.to_numpy()


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, w):
    m = len(y)
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-10, 1 - 1e-10)  # avoid log(0)
    loss = -(1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss


def gradient_descent_logistic(X, y, learning_rate, iterations):
    m, n = X.shape
    w = np.zeros(n)
    losses = {}

    for i in range(1, iterations + 1):
        p = sigmoid(X @ w)
        gradient = (1 / m) * (X.T @ (p - y))
        w = w - learning_rate * gradient

        if i in [10, 50, 100]:
            losses[i] = compute_loss(X, y, w)

    return w, losses


def predict_gd(X, w, threshold=0.5):
    probs = sigmoid(X @ w)
    return (probs >= threshold).astype(int)


learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    w, losses = gradient_descent_logistic(X_train_gd, y_train_gd, lr, 100)

    # training predictions
    y_train_pred_gd = predict_gd(X_train_gd, w)

    # testing predictions
    y_test_pred_gd = predict_gd(X_test_gd, w)

    # training metrics
    train_acc = accuracy_score(y_train_gd, y_train_pred_gd)
    train_prec = precision_score(y_train_gd, y_train_pred_gd)
    train_rec = recall_score(y_train_gd, y_train_pred_gd)
    train_f1 = f1_score(y_train_gd, y_train_pred_gd)

    # testing metrics
    test_acc = accuracy_score(y_test_gd, y_test_pred_gd)
    test_prec = precision_score(y_test_gd, y_test_pred_gd)
    test_rec = recall_score(y_test_gd, y_test_pred_gd)
    test_f1 = f1_score(y_test_gd, y_test_pred_gd)

    print(f"\nLearning Rate: {lr}")
    print("Cross-Entropy Loss:")
    print(f"  Iteration 10:  {losses[10]}")
    print(f"  Iteration 50:  {losses[50]}")
    print(f"  Iteration 100: {losses[100]}")

    print("Training Metrics at 100 iterations:")
    print(f"  Accuracy:  {train_acc}")
    print(f"  Precision: {train_prec}")
    print(f"  Recall:    {train_rec}")
    print(f"  F1 Score:  {train_f1}")

    print("Testing Metrics at 100 iterations:")
    print(f"  Accuracy:  {test_acc}")
    print(f"  Precision: {test_prec}")
    print(f"  Recall:    {test_rec}")
    print(f"  F1 Score:  {test_f1}")


# Compare with package logistic regression on scaled data
print("\nPackage Logistic Regression Comparison")

package_model = LogisticRegression(max_iter=1000)
package_model.fit(X_train_scaled, y_train_gd)

y_train_pkg = package_model.predict(X_train_scaled)
y_test_pkg = package_model.predict(X_test_scaled)

print("\nTraining Metrics (Package Model):")
print("  Accuracy:", accuracy_score(y_train_gd, y_train_pkg))
print("  Precision:", precision_score(y_train_gd, y_train_pkg))
print("  Recall:", recall_score(y_train_gd, y_train_pkg))
print("  F1 Score:", f1_score(y_train_gd, y_train_pkg))

print("\nTesting Metrics (Package Model):")
print("  Accuracy:", accuracy_score(y_test_gd, y_test_pkg))
print("  Precision:", precision_score(y_test_gd, y_test_pkg))
print("  Recall:", recall_score(y_test_gd, y_test_pkg))
print("  F1 Score:", f1_score(y_test_gd, y_test_pkg))

print("problem 3")
print("problem 3")
print("problem 3")
print("problem 3")
print("problem 3")
print("problem 3")

print("\nquestion 3")

# =========================
# Question 3
# Comparing classifiers
# =========================

# Use the same train/test split from Problem 1
# For kNN and LDA, scaling is helpful, so we use scaled features here
# Logistic regression will also use scaled features here for fairness

# -----------------------------------
# Part 1: Cross-validation for kNN
# -----------------------------------

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nkNN Cross-Validation Results")
best_k = None
best_cv_error = float("inf")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    cv_accuracy = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='precision')
    cv_recall = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='recall')

    avg_accuracy = cv_accuracy.mean()
    avg_error = 1 - avg_accuracy
    avg_precision = cv_precision.mean()
    avg_recall = cv_recall.mean()

    print(f"\nk = {k}")
    print(f"Average CV Accuracy:  {avg_accuracy}")
    print(f"Average CV Error:     {avg_error}")
    print(f"Average CV Precision: {avg_precision}")
    print(f"Average CV Recall:    {avg_recall}")

    if avg_error < best_cv_error:
        best_cv_error = avg_error
        best_k = k

print(f"\nBest k selected by cross-validation: {best_k}")
print(f"Lowest average CV error: {best_cv_error}")


# -----------------------------------
# Part 2: Train all 3 classifiers
# -----------------------------------

# Logistic Regression
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train)

# kNN with best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)


def evaluate_classifier(name, model, X_train_data, X_test_data, y_train_true, y_test_true):
    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    train_error = 1 - train_acc
    train_prec = precision_score(y_train_true, y_train_pred)
    train_rec = recall_score(y_train_true, y_train_pred)

    test_acc = accuracy_score(y_test_true, y_test_pred)
    test_error = 1 - test_acc
    test_prec = precision_score(y_test_true, y_test_pred)
    test_rec = recall_score(y_test_true, y_test_pred)

    print(f"\n{name}")
    print("Training Metrics:")
    print(f"  Accuracy:  {train_acc}")
    print(f"  Error:     {train_error}")
    print(f"  Precision: {train_prec}")
    print(f"  Recall:    {train_rec}")

    print("Testing Metrics:")
    print(f"  Accuracy:  {test_acc}")
    print(f"  Error:     {test_error}")
    print(f"  Precision: {test_prec}")
    print(f"  Recall:    {test_rec}")

    return {
        "train_accuracy": train_acc,
        "train_error": train_error,
        "train_precision": train_prec,
        "train_recall": train_rec,
        "test_accuracy": test_acc,
        "test_error": test_error,
        "test_precision": test_prec,
        "test_recall": test_rec
    }


log_results = evaluate_classifier(
    "Logistic Regression", log_model, X_train_scaled, X_test_scaled, y_train, y_test
)

lda_results = evaluate_classifier(
    "LDA", lda_model, X_train_scaled, X_test_scaled, y_train, y_test
)

knn_results = evaluate_classifier(
    f"kNN (k={best_k})", knn_model, X_train_scaled, X_test_scaled, y_train, y_test
)


# -----------------------------------
# Part 3: ROC curve + AUC for logistic regression
# -----------------------------------

y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)
auc_score = roc_auc_score(y_test, y_prob_log)

print("\nLogistic Regression ROC / AUC")
print("AUC:", auc_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Package ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------------
# Part 4: Manual ROC curve using thresholds
# -----------------------------------

manual_thresholds = np.arange(0, 1.01, 0.1)
manual_fpr = []
manual_tpr = []

for t in manual_thresholds:
    y_manual_pred = (y_prob_log >= t).astype(int)

    cm_manual = confusion_matrix(y_test, y_manual_pred)
    tn, fp, fn, tp = cm_manual.ravel()

    tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0

    manual_tpr.append(tpr_value)
    manual_fpr.append(fpr_value)

    print(f"\nThreshold = {t:.1f}")
    print(f"TPR: {tpr_value}")
    print(f"FPR: {fpr_value}")

plt.figure(figsize=(8, 6))
plt.plot(manual_fpr, manual_tpr, marker='o', label='Manual ROC (thresholds 0 to 1 step 0.1)')
plt.plot(fpr, tpr, label='Package ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Manual ROC vs Package ROC")
plt.legend()
plt.grid(True)
plt.show()

print("question 4")
print("question 4")
print("question 4")
print("question 4")
print("question 4")
print("question 4")

print("\nquestion 4")

# =========================
# Question 4
# Cross Validation
# =========================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Use full dataset for this problem
X_full = df.iloc[:, :-1].values
y_full = df.iloc[:, -1].values


def manual_k_fold_cv(model_class, X, y, k):
    n = len(X)

    # shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(n)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # split into k folds
    X_folds = np.array_split(X_shuffled, k)
    y_folds = np.array_split(y_shuffled, k)

    errors = []

    for i in range(k):
        # validation fold
        X_val = X_folds[i]
        y_val = y_folds[i]

        # training folds
        X_train_cv = np.concatenate([X_folds[j] for j in range(k) if j != i], axis=0)
        y_train_cv = np.concatenate([y_folds[j] for j in range(k) if j != i], axis=0)

        # scale inside each fold to avoid leakage
        scaler = StandardScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_val = scaler.transform(X_val)

        # train model
        model = model_class()
        model.fit(X_train_cv, y_train_cv)

        # predict and compute validation error
        y_val_pred = model.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_val_pred)
        fold_error = 1 - fold_accuracy
        errors.append(fold_error)

        print(f"Fold {i+1}/{k} validation error: {fold_error}")

    avg_error = np.mean(errors)
    return avg_error


# Logistic Regression CV
print("\nLogistic Regression Cross-Validation")
for k in [5, 10]:
    avg_error = manual_k_fold_cv(
        lambda: LogisticRegression(max_iter=2000),
        X_full,
        y_full,
        k
    )
    print(f"Average validation error for Logistic Regression with k={k}: {avg_error}")


# LDA CV
print("\nLDA Cross-Validation")
for k in [5, 10]:
    avg_error = manual_k_fold_cv(
        lambda: LinearDiscriminantAnalysis(),
        X_full,
        y_full,
        k
    )
    print(f"Average validation error for LDA with k={k}: {avg_error}")
