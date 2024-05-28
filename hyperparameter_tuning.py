# hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    return best_rf, grid_search.best_params_, grid_search.best_score_
