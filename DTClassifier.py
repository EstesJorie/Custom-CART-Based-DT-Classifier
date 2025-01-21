import pandas as pd
import numpy as np
import time
import psutil
import platform
import os
import csv
from tqdm.auto import tqdm

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
class DTClassifer:
    def __init__(self, min_Samples_Split=2, max_Depth=2, random_state=None):
        self.min_Samples_Split = min_Samples_Split
        self.max_Depth = max_Depth
        self.random_state = random_state
        self.root = None

    def mostCommonLabel(self, Y):
        Y = list(Y)
        return max(set(Y), key=Y.count)

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        try:
            data = np.concatenate((X, Y), axis=1)
        except ValueError as e:
            return
        
        self.root = self.buildTree(data, depth=0)  # Start building the tree from the root

    def buildTree(self, data, depth=0):
        X, Y = data[:, :-1], data[:, -1]
        num_samples, num_features = X.shape
        
        if num_samples < self.min_Samples_Split or depth >= self.max_Depth:
            leaf_value = self.mostCommonLabel(Y)  
            return Node(value=leaf_value)
        
        best_split = self.getBestSplit(data, num_samples, num_features)
        
        if best_split["info_gain"] > 0:
            left_subtree = self.buildTree(best_split["data_left"], depth + 1)
            right_subtree = self.buildTree(best_split["data_right"], depth + 1)
            
            return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.mostCommonLabel(Y)  
        return Node(value=leaf_value)

    def getBestSplit(self, data, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                data_left, data_right = self.split(data, feature_index, threshold)
                if len(data_left) > 0 and len(data_right) > 0:
                    y, left_y, right_y = data[:, -1], data_left[:, -1], data_right[:, -1]
                    curr_info_gain = self.informationGain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, data, feature_index, threshold):
        data_left = np.array([row for row in data if row[feature_index] <= threshold])
        data_right = np.array([row for row in data if row[feature_index] > threshold])
        return data_left, data_right

    def informationGain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.giniIndex(parent) - (weight_l * self.giniIndex(l_child) + weight_r * self.giniIndex(r_child))
        return gain

    def giniIndex(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def makePrediction(self, x, tree):
        if tree.value is not None:
            return tree.value  
        feature_val = x[tree.feature_index]
        
        if feature_val <= tree.threshold:
            return self.makePrediction(x, tree.left) 
        else:
            return self.makePrediction(x, tree.right)  

    def predict(self, X):
        predictions = [self.makePrediction(x, self.root) for x in X]
        return predictions

    def printTree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(f"Class: {tree.value}")
        else:
            print(f"X_{tree.feature_index} <= {tree.threshold} ? {tree.info_gain}")
            print(f"{indent}left:", end="")
            self.printTree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.printTree(tree.right, indent + indent)

class timeMemory:
    @staticmethod
    def memoryUsage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

    @staticmethod
    def trackTimeMemory(model, X_train, Y_train, X_test, Y_test):
        start = time.time()
        startMem = timeMemory.memoryUsage()
        
        if hasattr(model, 'fit'):
            tqdm.write(f"Training model.")  
            model.fit(X_train, Y_train)  

        tqdm.write(f"Making predictions.")  
        Y_pred = []
        for x in tqdm(X_test, desc="Predicting", unit="sample"):
            Y_pred.append(model.predict([x]))  
        Y_pred = [y[0] for y in Y_pred]  

        end = time.time()
        endMem = timeMemory.memoryUsage()

        timeTaken = end - start
        memoryUsed = endMem - startMem

        return timeTaken, memoryUsed  

    @staticmethod
    def saveToCSV(filename, data):
        fieldnames = ['min_samples_split', 'max_depth', 'time_taken', 'memory_used', 'accuracy', 'precision', 'recall', 'f1_score']
        
        write_header = not os.path.exists(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader() 
            
            for entry in data:
                for label, metrics in entry['class_report'].items():
                    if label != 'accuracy':
                        row = {
                            'min_samples_split': entry['min_samples_split'],
                            'max_depth': entry['max_depth'],
                            'time_taken': entry['time_taken'],
                            'memory_used': entry['memory_used'],
                            'accuracy': entry['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1-score']
                        }
                        writer.writerow(row)

    def getSysInfo(self):
        system_info = platform.uname()

        system_info= f"""System Information:
    System: {system_info.system}
    Node Name: {system_info.node}
    Release: {system_info.release}
    Version: {system_info.version}
    Machine: {system_info.machine}
    Processor: {system_info.processor}

    CPU Information:
    Processor: {platform.processor()}
    Physical Cores: {psutil.cpu_count(logical=False)}
    Logical Cores: {psutil.cpu_count(logical=True)}

    Memory Information:
    Total Memory: {psutil.virtual_memory().total} bytes
    Available Memory: {psutil.virtual_memory().available} bytes
    Used Memory: {psutil.virtual_memory().used} bytes
    Memory Utilization: {psutil.virtual_memory().percent}%
    """
        return system_info
