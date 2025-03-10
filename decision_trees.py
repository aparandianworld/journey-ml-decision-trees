import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Check missing values
print("Original data shape: ", df.shape)
print("Number of missing values: ", df.isnull().sum())
df = df.dropna(how="any")
print("Shape after removing missing values: ", df.shape)

# Data statistics
print("\nData statistics: ")
print(df.describe())

# Data preview
print("\nData preview: ")
print(df.head())

# Feature names and target vector
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
target_name = "target"

X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("\nFeature matrix shape: ", X.shape)
print("Target vector shape: ", y.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training and hyperparameter tuning
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)

# Make predictions
y_hat = model.predict(X_test)

# Model evaluation
print("\nModel evaluation: ")
print("Accuracy: ", accuracy_score(y_test, y_hat))
print("\nClassification report: ")
print(classification_report(y_test, y_hat))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, y_hat))

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(
    model, filled=True, feature_names=feature_names, class_names=iris.target_names
)
plt.show()
