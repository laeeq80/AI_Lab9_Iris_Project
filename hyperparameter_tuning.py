# hyperparameter_tuning.py

# Import necessary library: Import GridSearchCV from sklearn.model_selection.

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning for Random Forest
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30]
    }
    # Initialize GridSearchCV: Instantiate GridSearchCV with RandomForestClassifier, parameter grid,
    # 5-fold cross-validation, and accuracy as the scoring metric.
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    
    # Fit GridSearchCV: Fit the grid search to the training data.
    grid_search.fit(X_train, y_train)
    
    # Train with best parameters: Retrieve the best model and fit it to the training data.
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    return best_rf, grid_search.best_params_, grid_search.best_score_
