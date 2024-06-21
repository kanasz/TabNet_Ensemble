import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils import resample


class BaggingTabNet:
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
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_resampled, y_resampled = resample(X, y, replace=True, random_state=None)

            # Train TabNet model
            model = TabNetClassifier(n_a=self.n_a, n_d=self.n_d, n_steps=self.n_steps, gamma=self.gamma,
                                     verbose=0, lambda_sparse=self.lambda_sparse, momentum=self.momentum,
                                     n_shared=self.n_shared, n_independent=self.n_independent, seed=self.seed,
                                     device_name=self.device)
            model.fit(X_train=X_resampled, y_train=y_resampled, max_epochs=max_epochs, eval_metric=eval_metric,
                      loss_fn=loss_fn, patience=patience, batch_size=batch_size, drop_last=drop_last)

            self.models.append(model)

    def predict(self, X):
        # Collect predictions from all models
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        # Majority vote
        final_pred = np.round(np.mean(predictions, axis=1)).astype(int)

        return final_pred
