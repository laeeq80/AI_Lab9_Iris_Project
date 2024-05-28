# data_preprocessing.py

# Import necessary libraries: We import train_test_split for splitting the dataset
# and StandardScaler for feature scaling.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    # Separate features and target: X contains the features, and
    # y contains the target variable (species).
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Split data: train_test_split divides the data into training and testing sets (80% train, 20% test)
    # with a fixed random state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features: StandardScaler standardizes the features by removing
    # the mean and scaling to unit variance.
    # fit_transform is applied to the training set and transform to the test set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Returning training and testing datasest to the main
    return X_train, X_test, y_train, y_test
