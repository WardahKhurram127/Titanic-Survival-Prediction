import pandas as pd

df = pd.read_csv('data/titanic.csv')

df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

df.to_csv('data/titanic_cleaned.csv', index=False)
print("✅ Cleaned data saved to data/titanic_cleaned.csv")
