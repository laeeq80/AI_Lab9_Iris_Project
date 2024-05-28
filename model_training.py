# model_training.py
def train_models(models, X_train, y_train):
    for name, model in models.items():
        model.fit(X_train, y_train)
