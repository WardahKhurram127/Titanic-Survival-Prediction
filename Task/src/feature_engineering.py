import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data/titanic_cleaned.csv')

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

df.dropna(subset=["Age", "Fare"], inplace=True)

df["Age_Fare"] = df["Age"] * df["Fare"]
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Fare_Per_Person"] = df["Fare"] / df["Family_Size"]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["Age", "Fare"]])

poly_column_names = ["poly_" + name for name in poly.get_feature_names_out(["Age", "Fare"])]
poly_df = pd.DataFrame(poly_features, columns=poly_column_names)

df.reset_index(drop=True, inplace=True)
poly_df.reset_index(drop=True, inplace=True)

df = pd.concat([df, poly_df], axis=1)

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 35, 60, 100],
    labels=["Teen", "Young", "Adult", "Senior"]
)

df["Age_Group_Code"] = df["Age_Group"].map({
    "Teen": 0,
    "Young": 1,
    "Adult": 2,
    "Senior": 3
})

df.to_csv("data/titanic_with_features.csv", index=False)
print("✅ Feature-engineered dataset saved as 'titanic_with_features.csv'")
