import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from models.generic_model import GenericModel
from .dataset import Dataset
from .config import L2CatboostConfig


class L2CatboostModel(GenericModel):
    MODEL_NAME = 'l2_catboost'

    def __init__(self, config=L2CatboostConfig(), dataset=None):
        super().__init__()
        self.config = config
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = Dataset(config=config)

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        models = []
        classes_weights = [1.3, 2.0, 1., 2.8, 1., 2.1]
        for class_ix, class_name in enumerate(self.config.LIST_CLASSES):
            if X_val is not None:
                y_train_i = y_train[:, class_ix]
                y_val_i = y_val[:, class_ix]
                model = CatBoostClassifier(max_depth=3,
                                           colsample_bylevel=0.45,
                                           loss_function='Logloss',
                                           eval_metric='AUC',
                                           n_estimators=599,
                                           learning_rate=0.025,
                                           class_weights=[1, classes_weights[class_ix]],
                                           reg_lambda=0.35,
                                           bagging_temperature=0.35,
                                           use_best_model=True,
                                           random_seed=12345)

                model.fit(X_train, y_train_i, eval_set=(X_val, y_val_i))
                models.append(model)
        return models

    def predict(self, X, models, **kwargs):
        y_pred = np.zeros((X.shape[0], len(models)), dtype=np.float)
        for class_ix, model in enumerate(models):
            y_pred[:, class_ix] = model.predict_proba(X)[:,1]
        return y_pred
