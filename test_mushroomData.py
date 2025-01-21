import pytest
import os
import pandas as pd
from mushroomData import loadMushroomData, mushroomDescribe, checkMissingValues

def test_loadMushroomDataPrimary():
    filePath  = r'DATASETS\MushroomDataset\primary_data.csv'
    mushroomData = loadMushroomDataData(filePath)

    assert isinstance(mushroomData, pd.DataFrame)
    assert mushroomData.shape == (173, 20)

def test_loadMushroomDataSecondary():
    filePath  = r'DATASETS\MushroomDataset\secondary_data.csv'
    mushroomData = loadMushroomDataData(filePath)

    assert isinstance(mushroomData, pd.DataFrame)
    assert mushroomData.shape == (61069, 20)

    
def test_MushroomDescribePrimary():
    filePath  = r'DATASETS\MushroomDataset\primary_data.csv'
    mushroomData = loadMushroomDataData(filePath)
    mushroomDescription = mushroomDescribe(mushroomData)

    assert isinstance(mushroomDescription, pd.DataFrame)
    expectedColumns = [
    "cap-diameter","cap-shape","cap-surface","cap-color","does-bruise-bleed",
    "gill-attachment","gill-spacing","gill-color","stem-height","stem-width",
    "stem-root","stem-surface","stem-color","veil-type","veil-color",
    "has-ring","ring-type","spore-print-color","habitat","season"
]  
    assert list(mushroomDescription.columns) == expectedColumns
    
def test_MushroomDescribeSecondary():
    filePath  = r'DATASETS\MushroomDataset\secondary_data.csv'
    mushroomData = loadMushroomDataData(filePath)
    mushroomDescription = mushroomDescribe(mushroomData)

    assert isinstance(mushroomDescription, pd.DataFrame)
    expectedColumns = [
    "cap-diameter","cap-shape","cap-surface","cap-color","does-bruise-bleed",
    "gill-attachment","gill-spacing","gill-color","stem-height","stem-width",
    "stem-root","stem-surface","stem-color","veil-type","veil-color",
    "has-ring","ring-type","spore-print-color","habitat","season"
]  
    assert list(mushroomDescription.columns) == expectedColumns   
    
def test_ComparePrimaryAndSecondary():
    primaryFilePath  = r'DATASETS\MushroomDataset\primary_data.csv'
    secondaryFilePath  = r'DATASETS\MushroomDataset\secondary_data.csv'
    primaryMushroomData = loadMushroomData(primaryFilePath)
    secondaryMushroomData = loadMushroomData(secondaryFilePath)
    primaryMushroomDescribe = mushroomDescribe(primaryMushroomData)
    secondaryMushroomDescribe = mushroomDescribe(secondaryMushroomData)
    
    assert list(primaryMushroomDescribe.columns) == list(secondaryMushroomDescribe.columns)


def test_checkMissingValues():
    filePath  = r'DATASETS\MushroomDataset\primary_data.csv'
    mushroomData = loadMushroomDataData(filePath)
    missingValues = checkMissingValues(mushroomData)

    for missingCount in missingValues.items():
        assert missingCount == 0   
    
if __name__ == "__main__":
    pytest.main()