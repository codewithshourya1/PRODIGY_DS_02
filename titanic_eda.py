import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill missing 'Age' with the median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing 'Embarked' with mode
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Fill missing 'Fare' with the median
# Exploratory Data Analysis
# Correlation Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
# Summary of data
print("Data Summary:")
print(df.describe())
# Display first few rows to check the data after cleaning
print(df.head())
# Count the number of missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())
