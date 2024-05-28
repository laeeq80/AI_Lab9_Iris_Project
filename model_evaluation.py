# model_evaluation.py

# Import necessary metrics: Import accuracy_score, precision_score,
# recall_score, f1_score, and classification_report from sklearn.metrics.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Evaluate models: Loop through the dictionary, make predictions on the test
# data, and compute evaluation metrics for each model.
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Print results: Display accuracy, precision, recall, F1 score, and the
        # detailed classification report for each model.
        print(f"{name}:")
        print(f"Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1:.2f}")
        print(classification_report(y_test, y_pred))
