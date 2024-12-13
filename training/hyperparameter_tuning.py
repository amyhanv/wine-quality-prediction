from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
