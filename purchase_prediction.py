import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("purchase_training_data.csv")

X = data.drop(columns=["Purchased"])
y = data["Purchased"]

print("Features:", X.columns.tolist())
print("Target:", y.name)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(pd.DataFrame({
    "Train": y_train.value_counts(),
    "Test": y_test.value_counts()
}).sort_index())

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

print("Model training completed")
print("Classes:", model.classes_)


predictions = model.predict(X_test)
purchase_probability = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Purchase Probability (%)": (purchase_probability * 100).round(1)
})

print(results.head(10))
print("Total predictions:", len(results))


new_customers = pd.DataFrame({
    "Age":                [28, 42, 35, 23, 56, 31, 47, 39, 62, 25],
    "Income":             [120000, 65000, 82000, 32000, 95000, 54000, 110000, 72000, 88000, 48000],
    "Website_Visits":     [8, 2, 12, 4, 15, 6, 10, 3, 9, 5],
    "Time_Spent":         [25, 5, 45, 12, 55, 22, 38, 8, 32, 18],
    "Previous_Purchases": [3, 0, 6, 1, 8, 2, 5, 0, 4, 1],
    "Product_Views":      [12, 3, 24, 6, 28, 10, 20, 4, 16, 9]
})

new_results = new_customers.copy()
new_results["Predicted"] = model.predict(new_customers)
new_results["Purchase Probability (%)"] = (
    model.predict_proba(new_customers)[:, 1] * 100
).round(1)

print(new_results.to_string(index=False))


probabilities = model.predict_proba(new_customers)[:, 1]
default_predictions = model.predict(new_customers)

for threshold in [0.30, 0.50, 0.70]:
    labels = (probabilities >= threshold).astype(int)
    changed = (labels != default_predictions).sum()
    print(f"Threshold {threshold:.0%} | Predicted buyers: {labels.sum()} | Changed: {changed}")