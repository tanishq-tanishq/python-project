


# 1. DATA LOADING


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['species'] = iris.target

# Convert numeric target to flower names
df['species'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

# Inspect dataset
print(df.head())
print(df.info())


# 2. DATA CLEANING


# Check missing values
print(df.isnull().sum())

# Remove duplicates if any
df.drop_duplicates(inplace=True)


# 3. EXPLORATORY DATA ANALYSIS (EDA)


# Statistical summary
print(df.describe())

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64'])
cat_cols = df.select_dtypes(include=['object'])

# Histogram
df.hist(figsize=(8,6))
plt.show()

# Boxplot
sns.boxplot(data=df.drop('species', axis=1))
plt.show()

# Correlation
print(df.drop('species', axis=1).corr())

# Correlation heatmap
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.show()


# 4. DATA VISUALIZATION


# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Boxplot for petal length vs species
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.show()


# 5. PREDICTIVE MODELING (CLASSIFICATION)
#

# Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# 6. MODEL EVALUATION


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


