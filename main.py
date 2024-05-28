# main.py
from data_collection import load_data
from data_preprocessing import preprocess_data
from eda import perform_eda
from feature_selection import select_features
from model_selection import get_models
from model_training import train_models
from model_evaluation import evaluate_models
from hyperparameter_tuning import tune_hyperparameters

def main():
    # Step 1: Data Collection
    data = load_data()
    
    # Display data: Print the first few rows to inspect the dataset.
    print(data.head())

    # Step 2: Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3: Exploratory Data Analysis (EDA)
    perform_eda(data)

    # Step 4: Feature Selection
    feature_importance_df = select_features(X_train, y_train, data.columns[:-1])
    # Display feature importances: Print the sorted DataFrame.
    print(feature_importance_df)

    # Step 5: Model Selection
    models = get_models()

    # Step 6: Model Training
    train_models(models, X_train, y_train)

    # Step 7: Model Evaluation
    evaluate_models(models, X_test, y_test)

    # Step 8: Hyperparameter Tuning
    best_rf, best_params, best_score = tune_hyperparameters(X_train, y_train)
    
    # Print the best hyperparameters and the corresponding score.
    print("Best parameters:", best_params)
    print("Best score:", best_score)

if __name__ == "__main__":
    main()