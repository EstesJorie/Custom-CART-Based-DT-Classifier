import pytest
import os
import pandas as pd
from irisData import loadIrisData, irisDescribe, checkMissingValues

def test_loadIrisData():
    filePath = r'DATASETS\iris\iris.data'
    irisData = loadIrisData(filePath)

    assert isinstance(irisData, pd.DataFrame)
    assert irisData.shape == (150, 5)

    
def test_irisDescribe():
    filePath = r'DATASETS\iris\iris.data'
    irisData = loadIrisData(filePath)
    irisDescription = irisDescribe(irisData)

    assert isinstance(irisDescription, pd.DataFrame)
    expectedColumns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert list(irisDescription.columns) == expectedColumns
    

def test_checkMissingValues():
    filePath = r'DATASETS\iris\iris.data'
    irisData = loadIrisData(filePath)
    missingValues = checkMissingValues(irisData)

    for  missingCount in missingValues.items():
        assert missingCount == 0   
    
if __name__ == "__main__":
    pytest.main()