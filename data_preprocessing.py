# data_preprocessing.py

# Import necessary libraries: We import train_test_split for splitting the dataset
# and StandardScaler for feature scaling.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    # Separate features and target: X contains the features, and
    # y contains the target variable (species).
    X = data.drop('species', axis=1)   #axis=1 means to apply on coloums. axis=0 means to apply on rows.
    y = data['species']
    
    # Split data: train_test_split divides the data into training and testing sets
    # (80% train, 20% test) with a fixed random state (same as seed in C++)
    # for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features: StandardScaler standardizes the features by adjusting
    # the mean to 0 and scaling variance to 1.
    # fit_transform is applied to the training set and transform to the test set.
    # When you standardize features, you transform their values so that they have a mean of 0 and
    # a standard deviation of 1. This transformation doesn't change the shape of the distribution
    # of each feature; it simply shifts and scales the values to achieve the desired mean and
    # standard deviation. It increase the performance of the model.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Returning training and testing datasest to the main
    return X_train, X_test, y_train, y_test
    