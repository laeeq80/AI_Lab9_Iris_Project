# feature_selection.py

# Import necessary library: Import RandomForestClassifier from sklearn.ensemble.

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features(X_train, y_train, feature_names):
    # Fit model: Instantiate and fit a Random Forest model to the training data.
    model = RandomForestClassifier()
    model.fit(X_train, y_train) # We are training RF only for finding out important features
    
    # Feature importances: Retrieve the importance of each feature using model.feature_importances_.
    importances = model.feature_importances_
    
    # Create DataFrame: Create a DataFrame to display the feature importances.
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort features: Sort the features by importance in descending order.
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Retunr the features to the main
    return feature_importance_df
