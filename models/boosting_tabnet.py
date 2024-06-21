import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier


class BoostingTabNet:
    def __init__(self, n_estimators, learning_rate, n_d, n_a, n_steps, gamma, lambda_sparse, momentum, n_shared,
                 n_independent, device, seed=40):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  # 1.0
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.seed = seed
        self.models = []
        self.model_weights = []
        self.device = device

    def fit(self, X, y, eval_metric, loss_fn, max_epochs, patience, batch_size, drop_last):
        # Initialize sample weights
        sample_weights = np.ones(len(y)) / len(y)

        for _ in range(self.n_estimators):
            # Train TabNet model


            model = TabNetClassifier(n_a=self.n_a , n_d=self.n_d , n_steps=_+1, gamma=self.gamma,
                                     verbose=0, lambda_sparse=self.lambda_sparse, momentum=self.momentum,
                                     n_shared=self.n_shared, n_independent=self.n_independent, seed=self.seed,
                                     device_name=self.device)
            model.fit(X_train=X, y_train=y, max_epochs=max_epochs, weights=sample_weights, eval_metric=eval_metric,
                      loss_fn=loss_fn, patience=patience, batch_size=batch_size, drop_last=drop_last)
            # Predict training data
            y_pred = model.predict(X)

            # Compute the error rate
            incorrect = (y_pred != y)
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # Compute model weight
            model_weight = self.learning_rate * np.log((1 - error) / (error + 1e-10))
            self.model_weights.append(model_weight)

            # Update sample weights
            sample_weights = sample_weights * np.exp(model_weight * incorrect)
            sample_weights = sample_weights / np.sum(sample_weights)  # Normalize weights

            self.models.append(model)

    def predict(self, X):
        # Weighted sum of predictions from all models
        final_pred = np.zeros(len(X))
        for model, weight in zip(self.models, self.model_weights):
            final_pred += weight * model.predict_proba(X)[:, 1]

        # Convert probabilities to class labels
        return (final_pred >= 0.5).astype(int)
