import numpy as np
import lightgbm as lgb

from models.generic_model import GenericModel
from .dataset import Dataset
from .config import L2CatboostConfig


class L2CatboostModel(GenericModel):
    MODEL_NAME = 'l2_lgbm'

    def __init__(self, config=L2CatboostConfig(), dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = Dataset(config=config)

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        models = []
        for class_ix, class_name in enumerate(self.config.LIST_CLASSES):
            y_train_i = y_train[:, class_ix]

            model = lgb.LGBMClassifier(max_depth=2,
                                       metric="auc",
                                       n_estimators=350,
                                       num_leaves=20,
                                       boosting_type="gbdt",
                                       learning_rate=0.01,
                                       feature_fraction=0.4,
                                       colsample_bytree=0.8,
                                       bagging_fraction=0.6,
                                       bagging_freq=5,
                                       reg_lambda=0.25)

            model.fit(X_train, y_train_i)
            models.append(model)
        return models

    def predict(self, X, models, **kwargs):
        y_pred = np.zeros((X.shape[0], len(models)), dtype=np.float)
        for class_ix, model in enumerate(models):
            y_pred[:, class_ix] = model.predict_proba(X)[:, 1]
        return y_pred
