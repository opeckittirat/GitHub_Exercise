import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import CategoricalNB


# Load dataset
df = pd.read_csv("agaricus-lepiota.data", header=None)

columns = ["class"] + [f"feature_{i}" for i in range(1, df.shape[1])]
df.columns = columns


df_encoded = df.apply(lambda col: col.astype("category").cat.codes)

X = df_encoded.drop(columns=["class"])
y = df_encoded["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Custom naive bayes

def train_naive_bayes(X, y):
    classes = np.unique(y)
    priors = {}
    likelihoods = {}

    for c in classes:
        X_c = X[y == c]
        priors[c] = len(X_c) / len(X)

        likelihoods[c] = {}
        for col in X.columns:
            values = X[col].unique()
            likelihoods[c][col] = {}

            for v in values:
                # Laplace smoothing
                count = np.sum(X_c[col] == v)
                likelihoods[c][col][v] = (count + 1) / (len(X_c) + len(values))

    return priors, likelihoods


def predict_naive_bayes(X, priors, likelihoods):
    predictions = []

    for _, row in X.iterrows():
        class_probs = {}

        for c in priors:
            prob = np.log(priors[c])  # use log to prevent underflow

            for col in X.columns:
                val = row[col]
                prob += np.log(likelihoods[c][col].get(val, 1e-6))

            class_probs[c] = prob

        predictions.append(max(class_probs, key=class_probs.get))

    return np.array(predictions)


priors, likelihoods = train_naive_bayes(X_train, y_train)

y_pred_custom = predict_naive_bayes(X_test, priors, likelihoods)

# Metrics
custom_acc = accuracy_score(y_test, y_pred_custom)
custom_prec = precision_score(y_test, y_pred_custom)
custom_rec = recall_score(y_test, y_pred_custom)
custom_f1 = f1_score(y_test, y_pred_custom)

print("Custom Naive Bayes")
print("Accuracy:", round(custom_acc, 4))
print("Precision:", round(custom_prec, 4))
print("Recall:", round(custom_rec, 4))
print("F1 Score:", round(custom_f1, 4))
print()

nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)

y_pred_sklearn = nb_model.predict(X_test)

sk_acc = accuracy_score(y_test, y_pred_sklearn)
sk_prec = precision_score(y_test, y_pred_sklearn)
sk_rec = recall_score(y_test, y_pred_sklearn)
sk_f1 = f1_score(y_test, y_pred_sklearn)

print("Sklearn Naive Bayes")
print("Accuracy:", round(sk_acc, 4))
print("Precision:", round(sk_prec, 4))
print("Recall:", round(sk_rec, 4))
print("F1 Score:", round(sk_f1, 4))

probs = nb_model.predict_proba(X_test)

print("First 10 test points (Sklearn probabilities):")
for i in range(10):
    print(
        "P(Edible):", round(probs[i][0], 4),
        "P(Poisonous):", round(probs[i][1], 4)
    )
