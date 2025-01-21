import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from DTClassifier import DTClassifer 
from DTClassifier import timeMemory 

irisClasses = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
irisFilePath = r'./DATASETS/iris/iris.data'
irisData = pd.read_csv(irisFilePath, names=irisClasses, on_bad_lines='skip')

X = irisData.iloc[:, :-1].values
Y = irisData.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=36827814)


results = []

for min_samples_split in range(2, 16):
    for max_depth in range(1, 11):
        irisClassifier = DTClassifer(min_Samples_Split=min_samples_split, max_Depth=max_depth)
        timeTaken, memoryUsed = timeMemory.trackTimeMemory(irisClassifier, X_train, Y_train,  X_test, Y_test)
        print(f"Training Time: {timeTaken:.4f} seconds")
        print(f"Memory Usage: {memoryUsed:.4f} MB")
        #irisClassifier.printTree()
        Y_pred = irisClassifier.predict(X_test)
        Accuracy = accuracy_score(Y_test, Y_pred)
        classReport = classification_report(Y_test, Y_pred, output_dict=True)
        for label, metrics in classReport.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"Class: {label}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"{label.capitalize()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metrics:.4f}")

        data = {
            'min_samples_split': min_samples_split,  
            'max_depth': max_depth,
            'time_taken': timeTaken,
            'memory_used': memoryUsed,
            'accuracy': Accuracy,
            'class_report': classReport
        }
        
        results.append(data)

timeMemory.saveToCSV('irisClassifier.csv', results)

resultsSK = []

for min_samples_split in range(2, 16):
    for max_depth in range(1,10):
        irisSK = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, random_state=36827814)
        timeTaken, memoryUsed = timeMemory.trackTimeMemory(irisSK, X_train, Y_train, X_test, Y_test)
        print(f"Training Time: {timeTaken:.4f} seconds")
        print(f"Memory Usage: {memoryUsed:.4f} MB")

        Y_pred = irisSK.predict(X_test)
        Accuracy = accuracy_score(Y_test, Y_pred)
        classReport = classification_report(Y_test, Y_pred, output_dict=True)
        for label, metrics in classReport.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"Class: {label}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"{label.capitalize()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metrics:.4f}")

        skData = {
            'min_samples_split': min_samples_split,  
            'max_depth': max_depth,
            'time_taken': timeTaken,
            'memory_used': memoryUsed,
            'accuracy': Accuracy,
            'class_report': classReport
        }
        resultsSK.append(skData)

timeMemory.saveToCSV('irisSKLearn.csv', resultsSK)