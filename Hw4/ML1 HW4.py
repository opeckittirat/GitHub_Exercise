import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("spambase.data", header=None)

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


def get_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)

    return {
        "train_error": train_error,
        "test_error": test_error,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_auc": train_auc,
        "test_auc": test_auc
    }


# Part 1: Decision tree with information gain (entropy), no pruning
entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

entropy_results = get_metrics(entropy_tree, X_train, X_test, y_train, y_test)

print("Decision Tree with Information Gain (Entropy)")
for metric, value in entropy_results.items():
    print(metric + ":", round(value, 4))


print()


# Part 2: Decision tree with Gini index, no pruning
gini_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

gini_results = get_metrics(gini_tree, X_train, X_test, y_train, y_test)

print("Decision Tree with Gini Index")
for metric, value in gini_results.items():
    print(metric + ":", round(value, 4))


print()


# Part 3: Pruning using max_depth
depths = range(1, 21)
train_errors = []
test_errors = []

for depth in depths:
    pruned_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        random_state=42
    )

    results = get_metrics(pruned_tree, X_train, X_test, y_train, y_test)
    train_errors.append(results["train_error"])
    test_errors.append(results["test_error"])


best_depth = depths[np.argmin(test_errors)]
lowest_test_error = min(test_errors)

print("Best depth based on lowest testing error:", best_depth)
print("Lowest testing error:", round(lowest_test_error, 4))


plt.figure(figsize=(8, 5))
plt.plot(depths, train_errors, marker='o', label="Training Error")
plt.plot(depths, test_errors, marker='o', label="Testing Error")
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.title("Training and Testing Error vs Tree Depth")
plt.legend()
plt.grid(True)
plt.show()

# Q2
###
print("Question 2")
print("Question 2")
print("Question 2")
print("Question 2")
print("Question 2")


n_trees_list = [10, 50, 100, 500]
rf_results = []

for T in n_trees_list:
    rf_model = RandomForestClassifier(
        n_estimators=T,
        random_state=42
    )

    results = get_metrics(rf_model, X_train, X_test, y_train, y_test)

    rf_results.append({
        "T": T,
        "train_accuracy": results["train_accuracy"],
        "test_accuracy": results["test_accuracy"],
        "train_f1": results["train_f1"],
        "test_f1": results["test_f1"],
        "train_auc": results["train_auc"],
        "test_auc": results["test_auc"]
    })

    print("Random Forest with", T, "trees")
    print("train_accuracy:", round(results["train_accuracy"], 4))
    print("test_accuracy:", round(results["test_accuracy"], 4))
    print("train_f1:", round(results["train_f1"], 4))
    print("test_f1:", round(results["test_f1"], 4))
    print("train_auc:", round(results["train_auc"], 4))
    print("test_auc:", round(results["test_auc"], 4))
    print()


rf_results_df = pd.DataFrame(rf_results)

print("Random Forest Summary Table")
print(rf_results_df.round(4))
print()


rf_500 = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf_500.fit(X_train, y_train)

feature_importances = rf_500.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("Top 10 Most Important Features")
print(importance_df.head(10).round(4))


# Plot top 20 feature by importance lol
top_n = 20
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.bar(range(top_n), top_features["Importance"])
plt.xticks(range(top_n), top_features["Feature"], rotation=90)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Top 20 Feature Importances from Random Forest")
plt.tight_layout()
plt.show()

#q3
print("Question 3")
print("Question 3")
print("Question 3")
print("Question 3")
print("Question 3")
print("Question 3")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve

ada_trees_list = [10, 50, 100, 500]
ada_results = []

for T in ada_trees_list:
    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=T,
        random_state=42
    )

    results = get_metrics(ada_model, X_train, X_test, y_train, y_test)

    ada_results.append({
        "T": T,
        "train_accuracy": results["train_accuracy"],
        "test_accuracy": results["test_accuracy"],
        "train_f1": results["train_f1"],
        "test_f1": results["test_f1"],
        "train_auc": results["train_auc"],
        "test_auc": results["test_auc"]
    })

    print("AdaBoost with", T, "base classifiers")
    print("train_accuracy:", round(results["train_accuracy"], 4))
    print("test_accuracy:", round(results["test_accuracy"], 4))
    print("train_f1:", round(results["train_f1"], 4))
    print("test_f1:", round(results["test_f1"], 4))
    print("train_auc:", round(results["train_auc"], 4))
    print("test_auc:", round(results["test_auc"], 4))
    print()


ada_results_df = pd.DataFrame(ada_results)

print("AdaBoost Summary Table")
print(ada_results_df.round(4))
print()


comparison_df = rf_results_df.merge(ada_results_df, on="T", suffixes=("_rf", "_ada"))

print("Random Forest vs AdaBoost Comparison")
print(comparison_df.round(4))
print()


dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_probs = dt_model.predict_proba(X_test)[:, 1]

rf_100_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_100_model.fit(X_train, y_train)
rf_100_probs = rf_100_model.predict_proba(X_test)[:, 1]

ada_100_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    random_state=42
)
ada_100_model.fit(X_train, y_train)
ada_100_probs = ada_100_model.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_100_probs)
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_100_probs)

auc_dt = roc_auc_score(y_test, dt_probs)
auc_rf = roc_auc_score(y_test, rf_100_probs)
auc_ada = roc_auc_score(y_test, ada_100_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest 100 Trees (AUC = {auc_rf:.4f})")
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost 100 Trees (AUC = {auc_ada:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True)
plt.show()

