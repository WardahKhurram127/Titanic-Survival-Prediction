import pandas as pd

# Load data
df = pd.read_csv('data/titanic.csv')

# Drop unneeded columns
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Save cleaned data
df.to_csv('data/titanic_cleaned.csv', index=False)
print("✅ Cleaned data saved to data/titanic_cleaned.csv")
