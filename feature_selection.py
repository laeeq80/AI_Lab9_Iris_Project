# feature_selection.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features(X_train, y_train, feature_names):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    return feature_importance_df
