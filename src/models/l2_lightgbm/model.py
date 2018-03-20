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
        classes_weights = [1, 1.5, 1, 2.5, 1, 1.5]

        for class_ix, class_name in enumerate(self.config.LIST_CLASSES):
            y_train_i = y_train[:, class_ix]

            model = CatBoostClassifier(max_depth=2,
                                       colsample_bylevel=0.8,
                                       loss_function='Logloss',
                                       eval_metric='AUC',
                                       n_estimators=250,
                                       learning_rate=0.01,
                                       class_weights=[1, classes_weights[class_ix]],
                                       reg_lambda=0.028)

            model.fit(X_train, y_train_i)
            models.append(model)
        return models

    def predict(self, X, models, **kwargs):
        y_pred = np.zeros((X.shape[0], len(models)), dtype=np.float)
        for class_ix, model in enumerate(models):
            y_pred[:, class_ix] = model.predict_proba(X)[:,1]
        return y_pred
