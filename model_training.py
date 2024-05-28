# model_training.py

# Train models: Loop through the dictionary and train each model
# using fit on the training data (X_train, y_train).
def train_models(models, X_train, y_train):
    for name, model in models.items():
        model.fit(X_train, y_train)
