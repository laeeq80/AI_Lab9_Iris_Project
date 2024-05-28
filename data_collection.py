# data_collection.py
from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data
