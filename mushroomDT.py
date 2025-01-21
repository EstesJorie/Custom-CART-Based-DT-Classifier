import pandas as pd
import matplotlib as plt
import numpy as np
import scipy as sp
#import seaborn as sns
import time
from tqdm import tqdm

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from DTClassifier import DTClassifer 
from DTClassifier import timeMemory 

mushroomClasses = [
    "class", "cap-diameter", "cap-shape", "cap-surface", "cap-color", 
    "does-bruise-bleed", "gill-attachment", "gill-spacing", "gill-color", 
    "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color", 
    "veil-type", "veil-color", "has-ring", "ring-type", "spore-print-color", 
    "habitat", "season"
]  
mushroomFilepath = r'./DATASETS/MushroomDataset/secondary_data.csv'
mushroomData = pd.read_csv(mushroomFilepath, names=mushroomClasses, on_bad_lines='skip', sep=';')

# Label encode the 'class' column (target)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
label_encoders["class"] = LabelEncoder()
mushroomData["class"] = label_encoders["class"].fit_transform(mushroomData["class"])

# Define categorical feature columns for one-hot encoding
categorical_features = [
    "cap-shape", "cap-surface", "cap-color", "does-bruise-bleed", 
    "gill-attachment", "gill-spacing", "gill-color", "stem-root", 
    "stem-surface", "stem-color", "veil-type", "veil-color", 
    "has-ring", "ring-type", "spore-print-color", "habitat", "season"
]

# Define features and target
X = mushroomData.drop(columns=["class"])  # Keep X as DataFrame, not NumPy array
Y = mushroomData["class"].values  # Target ('class')

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=36827814)

transformer = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)], 
    remainder="passthrough"  # Keep numerical columns unchanged
)

results = []

for min_samples_split in range (2, 16):
    for max_depth in range(1, 11):
        mushroomClassifier = Pipeline(steps=[("preprocessor", transformer), 
                                            ("classifier", DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, random_state=36827814))])
        start = time.time()
        startMem = timeMemory.memoryUsage()
        tqdm.write("Training model.")
        for _ in tqdm(range(1), desc="Training", unit="epoch"):
            mushroomClassifier.fit(X_train, Y_train)
        end = time.time()
        endMem = timeMemory.memoryUsage()

        timeTaken = end - start
        memoryUsed = endMem - startMem

        print(f"Training Time: {timeTaken:.4f} seconds")
        print(f"Memory Usage: {memoryUsed:.4f} MB")

        tqdm.write("Making predictions.")
        Y_pred = mushroomClassifier.predict(X_test)

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
            'min_samples_split': max_depth,  
            'max_depth': max_depth,
            'time_taken': timeTaken,
            'memory_used': memoryUsed,
            'accuracy': Accuracy,
            'class_report': classReport
        }

        results.append(data)

timeMemory.saveToCSV('mushroomClassifier.csv', results)

resultsSK = []

for min_samples_split in range(2, 16):
    for max_depth in range(1, 11):
        mushroomSK = Pipeline(steps=[("preprocessor", transformer), 
                                    ("classifier", DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, random_state=36827814))])

        startMem_sk = timeMemory.memoryUsage()
        start_sk = time.time()
        mushroomSK.fit(X_train, Y_train)

        Y_pred_sk = mushroomSK.predict(X_test)
        end_sk = time.time()
        endMem_sk = timeMemory.memoryUsage()

        timeTaken_sk = end_sk - start_sk
        memoryUsed_sk = endMem_sk - startMem_sk

        Accuracy_sk = accuracy_score(Y_test, Y_pred_sk)
        classReport_sk = classification_report(Y_test, Y_pred_sk, output_dict=True)

        cv_scores = cross_val_score(mushroomSK, X, Y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

        for label, metrics in classReport_sk.items():
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
            'time_taken': timeTaken_sk,
            'memory_used': memoryUsed_sk,
            'accuracy': Accuracy_sk,
            'class_report': classReport_sk
        }

        resultsSK.append(skData)

timeMemory.saveToCSV('mushroomSKLearn.csv', resultsSK)