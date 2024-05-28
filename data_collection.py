# data_collection.py

# Import necessary libraries: We import the load_iris function
# from sklearn. datasets and pandas for data manipulation.
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset: load_iris() loads the dataset into a dictionary-like object.
def load_data():
    iris = load_iris()
    
    # Create a DataFrame: We convert the data into a pandas DataFrame for
    # easier handling and label the columns with feature names.
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Add target variable: Add the target variable (species) to the DataFrame.
    data['species'] = iris.target
    
    # Return Data to the main function
    return data
