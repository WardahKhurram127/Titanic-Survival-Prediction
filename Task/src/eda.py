import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/titanic_cleaned.csv')

# Plot 1: Survival Count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.savefig('outputs/plots/survival_count.png')
plt.clf()

# Plot 2: Age distribution
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.savefig('outputs/plots/age_distribution.png')
plt.clf()

# Plot 3: Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig('outputs/plots/correlation.png')

print("✅ EDA plots saved to outputs/plots/")
