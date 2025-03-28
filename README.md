# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Necessary Libraries and Load Data
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix
   
## Program:
## Program to implement the prediction of iris species using SGD Classifier.
### Developed by: Prasanna A
### RegisterNumber:  23005675

```
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)

y = df['target']

Load the Iris dataset

Create a Pandas DataFrame

Display the first few rows of the dataset

Split the data into features (X) and target (y)

Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

print(cm)
```
## Output:
![prediction of iris species using SGD Classifier](sam.png)
![4321](https://github.com/user-attachments/assets/5351a282-1e57-4e87-8c7f-1fa1a1f34372)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
