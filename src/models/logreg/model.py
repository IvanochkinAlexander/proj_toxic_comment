import numpy as np
from sklearn.linear_model import LogisticRegression
from models.generic_model import GenericModel
from models.generic_config import GenericConfig

from .dataset import Dataset

from .config import LogisticRegressionConfig

class LogisticRegressionModel(GenericModel):
    MODEL_NAME = 'lr'

    def __init__(self, config=LogisticRegressionConfig(), dataset=None, use_json=True):
        super().__init__()
        self.config = config
        if dataset is not None:
            print('Using pretrained dataset.')
            self.dataset = dataset
        else:
            self.dataset = Dataset(config=config, use_json=use_json)

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        if X_val is not None:
            return NotImplementedError('No validation is possible in fit for now.')

        models = []

        for class_ix, class_name in enumerate(self.config.LIST_CLASSES):
            y_train_i = y_train[:, class_ix]

            model = LogisticRegression(C=0.1, solver='sag', n_jobs=-1)

            model.fit(X_train, y_train_i)
            models.append(model)
        return models

    def predict(self, X, models, **kwargs):
        y_pred = np.zeros((X.shape[0], len(models)), dtype=np.float)
        for class_ix, model in enumerate(models):
            y_pred[:, class_ix] = model.predict_proba(X)[:, 1]
        return y_pred
