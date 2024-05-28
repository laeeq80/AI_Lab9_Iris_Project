# model_selection.py

# Import necessary libraries: Import LogisticRegression, RandomForestClassifier, and SVC from sklearn.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models():
    # Initialize models: Create a dictionary of models to compare, with names as
    # keys and model instances as values.
    # Set a higher max_iter for LogisticRegression to ensure convergence.
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC()
    }
    
    # Return Model to the main
    return models
