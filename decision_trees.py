import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target

# Check missing values
print("Original data shape: ", df.shape)
print("Number of missing values: ", df.isnull().sum())
df = df.dropna(how = 'any')
print("Shape after removing missing values: ", df.shape)

# Data statistics
print("\nData statistics: ")
print(df.describe())

# Data preview
print("\nData preview: ")
print(df.head())

# Feature names and target vector
feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
target_name = "target"

X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("\nFeature matrix shape: ", X.shape)
print("Target vector shape: ", y.shape)

# Split into train and test
