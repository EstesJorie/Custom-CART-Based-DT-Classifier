import pandas as pd 
import os

'''
Functions for PYTEST
'''

def loadIrisData(filePath):
    irisClasses = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    try:
        irisData = pd.read_csv(filePath, names=irisClasses)
        return irisData
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filePath}")
    
def irisDescribe(irisData):
    return irisData.describe()

def checkMissingValues(irisData):
    return irisData.isnull().sum()