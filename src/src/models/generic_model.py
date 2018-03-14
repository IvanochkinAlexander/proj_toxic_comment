import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime

from ..utils import metrics

class GenericModel:
    MODEL_NAME = 'generic_model'

    def __init__(self):
        self.config = Config
        self.experiment_id = f"{self.model_name}_{datetime.now().isoformat('-', timespec='minutes')}"
        self.trained_models = {}  # {'fold1': model, 'fold2': model, ...}

    def display_info(self):
        """Display model info collected from the config file."""
        print(self.config.display_info())

    def build_model(self):
        """Generic process for model creation.

        Returns:
        Initialized model.
        """
        raise NotImplementedError

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        """Fit a desired model.

        Returns:
        Trained model.
        """
        raise NotImplementedError

    def predict(self, X, model=None, **kwargs):
        if model is not None:
            y_pred = model.predict(X, **kwargs)
            return y_pred
        raise NotImplementedError

    def evaluate(self, X_val, y_val, model=None):
        if model is not None:
            y_pred = self.predict(X_val, model=model)
            logloss = log_loss(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred)
            return y_pred, logloss, auc
        return NotImplementedError

    def cross_validate(self, X=None, y=None):
        """Cross validate and print the average score as well as output averaged predictions.
        """
        kfold = KFold(n_splits=self.config.CV_NSPLITS, random_state=self.config.CV_RANDOM_STATE)
        oof = np.zeros_like(X)
        oof_loglosses, oof_aucs = [], []

        for i, (tr_ix, te_ix) in enumerate(kfold.split(X)):
            X_train, X_val = X[tr_ix], X[te_ix]
            y_train, y_val = y[tr_ix], y[te_ix]

            X_tr, X_foldval, y_tr, y_foldval = train_test_split(
                X_train, y_train, test_size=0.1,
                random_state=self.config.CV_RANDOM_STATE
            )

            model = self.fit(X_train, X_foldval, y_train, y_foldval)
            oof_kfold, oof_logloss, oof_auc = self.evaluate(X_val, y_val, model=model)
            oof_loglosses.append(oof_logloss)
            oof_aucs.append(oof_auc)

            oof[te_ix] = oof_kfold
            model_name = f'fold_{i}'
            self.trained_models[model_name] = model

        self._generate_cv_results(oof, oof_loglosses, oof_aucs)

    def crossval_inference(self):
        """Inference pipeline powered by cross validation. Early stopping may or may not
        be included, final crossval score is always printed.
        """
        pass

    def _generate_cv_results(self, oof, oof_loglosses, oof_aucs):
        average_logloss = np.sum(oof_loglosses) / len(oof_loglosses)
        average_auc = np.sum(oof_aucs) / len(oof_aucs)

        summary_filename = f'{self.experiment_id}_summary.txt'
        oof_filename = f'{self.experiment_id}.csv'
        summary_file_path = os.path.join(config.CV_OUTPUT_DIR, summary_filename)
        oof_filename = os.path.join(config.CV_OUTPUT_DIR, oof_filename)

        summary_model_info = self.config.display_info()
        summary_result_metric = "Average logloss: {:.5f}, average OOF ROC AUC: {:.5f}".format(average_logloss, average_auc)
        summary = '{}\n\n{}'.format(summary_result_metric, summary_model_info)

        with open(summary_filename, 'w') as f:
            f.write(summary)

        oof_file = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        oof_file[LIST_CLASSES] = oof
        oof_file.to_csv(oof_filename, index=False)

    def get_roc_auc_scorer(self):
        return metrics.RocAucEvaluation
