import json
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import StepLR

from base_functions import get_loss
from constants import LossFunction


class OCBaggingTabnetEnsemble:

    def __init__(self, config_files, cls_num_list, device):
        self.config_files = config_files
        self.cls_num_list = cls_num_list
        self.device = device
        self.loss_functions = []
        self.models = []
        self.load_models()



    def get_optimizer_fn(self, optimizer_name):
        if optimizer_name == "adam":
            from torch.optim import Adam
            return Adam
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_scheduler_fn(self, scheduler_name):
        if scheduler_name == "StepLR":
            return StepLR
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def load_models(self):
        for config_file in self.config_files:
            with open(config_file, 'r') as file:
                config = json.load(file)

            self.loss_functions.append(config["loss_function"])
            model = TabNetClassifier(
                verbose=0,
                n_d=config["n_d"],
                n_a=config["n_a"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                #cat_idxs=config["cat_idxs"],
                #cat_dims=config["cat_dims"],
                #cat_emb_dim=config["cat_emb_dim"],
                n_independent=config["n_independent"],
                n_shared=config["n_shared"],
                momentum=config["momentum"],
                #clip_value=config["clip_value"],
                lambda_sparse=config["lambda_sparse"],
                #optimizer_fn=self.get_optimizer_fn(config["optimizer_fn"]),
                #optimizer_params=config["optimizer_params"],
                #scheduler_params=config["scheduler_params"],
                #scheduler_fn=self.get_scheduler_fn(config["scheduler_fn"]),
                #mask_type=config["mask_type"],
                device_name=self.device
            )

            self.models.append(model)

    def fit(self, X, y, eval_metric, max_epochs, patience, batch_size, drop_last, cls_num_list, solution):
        self.load_models()
        # Train each model on a different bootstrap sample

        for idx, model in enumerate(self.models):
            loss_fn = get_loss(LossFunction(self.loss_functions[idx]), solution[8:], cls_num_list, self.device)
            model.fit(X, y,
                      #eval_set=[(X, y)],
                      eval_metric=eval_metric,
                      loss_fn=loss_fn,
                      max_epochs=max_epochs,
                      patience=patience,
                      batch_size=batch_size,
                      drop_last=drop_last)

    def bagging_predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_test)

        # Majority voting
        final_predictions = np.round(np.mean(predictions, axis=1)).astype(int)
        return final_predictions

    def predict(self, X_test):
        return self.bagging_predict(X_test)

# Usage example:
# config_files = ['tabnet_config_1.json', 'tabnet_config_2.json', 'tabnet_config_3.json']
# ensemble = TabnetEnsemble(config_files)
# X, y = load_data()  # Define this function as per your data loading logic
# ensemble.fit(X, y)
# X_test = np.random.rand(200, 20)  # Dummy test feature matrix
# final_predictions = ensemble.predict(X_test)
# print(final_predictions)
