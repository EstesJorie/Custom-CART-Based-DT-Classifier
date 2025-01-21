import pandas as pd 
import os

'''
Functions for PYTEST
'''

def loadMushroomData(filePath):
    mushroomClasses = [
    "cap-diameter","cap-shape","cap-surface","cap-color","does-bruise-bleed",
    "gill-attachment","gill-spacing","gill-color","stem-height","stem-width",
    "stem-root","stem-surface","stem-color","veil-type","veil-color",
    "has-ring","ring-type","spore-print-color","habitat","season"]   
try:
    mushroomData = pd.read_csv(filePath, names=mushroomClasses, on_bad_lines='skip', sep=';')
    return mushroomData
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {filePath}")
    
def mushroomDescribe(mushroomData):
    return mushroomData.describe()

def checkMissingValues(mushroomData):
    return mushroomData.isnull().sum()