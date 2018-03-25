import numpy as np
from wordbatch.models import FM_FTRL
from models.ftrl.config import FTRLConfig
from models.generic_model import GenericModel
from .dataset import Dataset


class FTRLModel(GenericModel):
    MODEL_NAME = 'ftrl'

    def __init__(self, config=FTRLConfig(), use_json=True):
        super().__init__()
        self.config = config
        self.dataset = Dataset(config=config, use_json=use_json)

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        if X_val is not None:
            return NotImplementedError('No validation is possible in fit for now.')

        models = []

        for class_ix, class_name in enumerate(self.config.LIST_CLASSES):
            y_train_i = y_train[:, class_ix]

            model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=30.0,
                          D=X_train.shape[1], alpha_fm=0.1,
                          L2_fm=0.5, init_fm=0.01,
                          D_fm=200, e_noise=0.0, iters=self.config.ITERATIONS,
                          inv_link="identity", threads=30)

            model.fit(X_train, y_train_i)
            models.append(model)
        return models

    def predict(self, X, models, **kwargs):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        y_pred = np.zeros((X.shape[0], len(models)), dtype=np.float)
        for class_ix, model in enumerate(models):
            y_pred[:, class_ix] = model.predict(X)
        return y_pred
