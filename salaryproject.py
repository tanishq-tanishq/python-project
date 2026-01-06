

# 1. DATA LOADING

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset using Pandas
df = pd.read_csv("Salary_Data.csv")

# Inspect first few rows
print(df.head())

# Check shape and column types
print(df.info())


# 2. DATA CLEANING


# Check missing values
print(df.isnull().sum())

# Handle missing values
df = df.dropna()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert data types
df['YearsExperience'] = df['YearsExperience'].astype(float)
df['Salary'] = df['Salary'].astype(float)


# 3. EXPLORATORY DATA ANALYSIS 

# Statistical summary
print(df.describe())

# Identify numerical and categorical variables
num_cols = df.select_dtypes(include=['int64', 'float64'])
cat_cols = df.select_dtypes(include=['object'])

# Histogram (distribution)
df.hist(figsize=(6,4))
plt.show()

# Boxplot
sns.boxplot(data=df)
plt.show()

# Correlation
print(df.corr())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# 4. DATA VISUALIZATION


# Scatter plot
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# Line plot
plt.plot(df['YearsExperience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Line Plot")
plt.show()

# Bar chart
sns.barplot(x='YearsExperience', y='Salary', data=df)
plt.show()


# 5. PREDICTIVE MODELING (REGRESSION)


# Split features and target
X = df[['YearsExperience']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# 6. MODEL EVALUATION


# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)



