'''
Kittirat T
ML Final Project
'''

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


df = pd.read_csv("AB_NYC_2019.csv")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# Remove outliers
df = df[(df["price"] > 0) & (df["price"] <= 500)]

feature_cols = [
    "neighbourhood_group",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

X = df[feature_cols]
y = df["price"]

categorical_features = ["neighbourhood_group", "room_type"]
numeric_features = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

numeric_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=8),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, random_state=42, max_depth=20, n_jobs=-1
    ),
}

for model_name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"{model_name}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2:  {r2:.4f}")
    print()

plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=50, color="#0d7377")
plt.xlabel("Price")
plt.ylabel("Number of Listings")
plt.title("Distribution of Airbnb Prices")
plt.show()

avg_price_borough = df.groupby("neighbourhood_group")["price"].mean()

plt.figure(figsize=(8,5))
avg_price_borough.plot(kind="bar", color="#0d7377")
plt.xlabel("Borough")
plt.ylabel("Average Price")
plt.title("Average Price by Borough")
plt.xticks(rotation=0)
plt.show()

avg_price_room = df.groupby("room_type")["price"].mean()

plt.figure(figsize=(8,5))
avg_price_room.plot(kind="bar", color="#0d7377")
plt.xlabel("Room Type")
plt.ylabel("Average Price")
plt.title("Average Price by Room Type")
plt.xticks(rotation=15)
plt.show()

sample_df = df.sample(3000, random_state=42)

plt.figure(figsize=(8,5))
plt.scatter(sample_df["availability_365"], sample_df["price"], alpha=0.3, color="#0d7377")
plt.xlabel("Availability (days/year)")
plt.ylabel("Price")
plt.title("Availability vs Price")
plt.show()

results = []

for model_name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([model_name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print(results_df)

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["RMSE"], color="#0d7377")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Model Comparison (Lower is Better)")
plt.xticks(rotation=15)
plt.show()

model = RandomForestRegressor(n_estimators=150, random_state=42)
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipe.fit(X_train, y_train)

ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_features = ohe.get_feature_names_out(categorical_features)

all_features = np.concatenate([numeric_features, cat_features])

importances = pipe.named_steps["model"].feature_importances_

feat_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_df.head(10))

example = X_test.iloc[[0]]
predicted_price = pipe.predict(example)

print("Predicted:", predicted_price)
print("Actual:", y_test.iloc[0])

plt.figure(figsize=(8,5))

plt.scatter(y_test, preds, alpha=0.3, color="#0d7377")

plt.scatter([60], [71.66], marker='x', s=100, color="#0d7377")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.plot([0, 500], [0, 500])

plt.show()

# ROC
threshold = y_train.median()
y_test_class = (y_test > threshold).astype(int)

fpr, tpr, _ = roc_curve(y_test_class, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="#0d7377", linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# cross validation

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=150, random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")

rmse_scores = -scores

print("Cross validation RMSE:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())