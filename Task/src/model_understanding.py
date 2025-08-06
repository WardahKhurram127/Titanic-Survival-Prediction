import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("data/titanic_with_features.csv")

target = "Survived"
features = [
    "Age", "Fare", "Age_Fare", "Family_Size", "Fare_Per_Person",
    "poly_Age", "poly_Fare", "poly_Age^2", "poly_Age Fare", "poly_Fare^2",
    "Age_Group_Code"
]

X = df[features]
y = df[target]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_val, y_train_val)

print("\n✅ Best Hyperparameters Found:")
print(grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_test_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print("\n📊 Test Set Evaluation with Tuned Random Forest:")
print(f"Accuracy:  {round(acc, 4)}")
print(f"Precision: {round(prec, 4)}")
print(f"Recall:    {round(rec, 4)}")
print(f"F1 Score:  {round(f1, 4)}")
print(f"\nConfusion Matrix:\n{cm}")

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [features[i] for i in indices[:10]]
top_importances = importances[indices[:10]]

plt.figure(figsize=(10, 6))
plt.bar(top_features, top_importances)
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_rf,
    X=X_train_val,
    y=y_train_val,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, val_mean, label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
