# hyperparameter_tuning.py

# Import necessary library: Import GridSearchCV from sklearn.model_selection.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30]
    }
    # Initialize GridSearchCV: Instantiate GridSearchCV with RandomForestClassifier, parameter grid,
    # 5-fold cross-validation, and accuracy as the scoring metric.
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    
    # Fit GridSearchCV: Fit the grid search to the training data.
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_rf = grid_search.best_estimator_
    return best_rf, best_params, best_score
