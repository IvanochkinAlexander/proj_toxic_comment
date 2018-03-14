import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime

from .generic_config import GenericConfig
from utils import metrics
from utils.send_to_telegram import send_to_telegram


# def send_to_telegram(msg):
    # pass


class GenericModel:
    MODEL_NAME = 'generic_model'

    def __init__(self):
        self.config = GenericConfig()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.getLogger().info(f'{self.experiment_id} Initializing with config {self.config.display_info()}.')
        self.trained_models = {}  # {'fold1': model, 'fold2': model, ...}

    @property
    def experiment_id(self):
        return f"{self.config.MODEL_PREFIX}_{self.MODEL_NAME}_{self.timestamp}"

    def display_info(self):
        """Display model info collected from the config file."""
        print(self.config.display_info())

    def _build_model(self):
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

    def predict(self, X, model, **kwargs):
        if model is not None:
            y_pred = model.predict(X, **kwargs)
            return y_pred
        raise NotImplementedError

    def evaluate(self, X_val, y_val, model):
        y_pred = self.predict(X_val, model)
        logloss = log_loss(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)
        return y_pred, logloss, auc

    def cross_validate(self, X=None, y=None):
        """Cross validate and print the average score as well as output averaged predictions.
        """
        kfold = KFold(n_splits=self.config.CV_NSPLITS, random_state=self.config.CV_RANDOM_STATE)
        oof = np.zeros_like(y, dtype=np.float)
        oof_loglosses, oof_aucs = [], []

        cv_start_msg = "*** CV STARTED ***\n\nMODEL:{}\nCONFIG:{}".format(
            self.experiment_id, self.config.display_info()
        )
        send_to_telegram(cv_start_msg)

        for i, (tr_ix, te_ix) in enumerate(kfold.split(X)):
            X_train, X_val = X[tr_ix], X[te_ix]
            y_train, y_val = y[tr_ix], y[te_ix]

            X_tr, X_foldval, y_tr, y_foldval = train_test_split(
                X_train, y_train, test_size=0.1,
                random_state=self.config.CV_RANDOM_STATE
            )
            fold_start_msg = f'MODEL: {self.experiment_id}\nCV FOLD {i + 1} / {self.config.CV_NSPLITS}'
            logging.getLogger().info(fold_start_msg)
            send_to_telegram(fold_start_msg)

            if self.config.FOLD_VAL_SPLIT is True:
                model = self.fit(X_tr, X_foldval, y_tr, y_foldval)
            else:
                model = self.fit(X_train=X_train, y_train=y_train)
            oof_kfold, oof_logloss, oof_auc = self.evaluate(X_val, y_val, model)

            fold_end_msg = 'MODEL: {}\nCV FOLD {} / {}\nLOGLOSS: {:.5f}, AUC: {:.5f}'.format(
                self.experiment_id, i, self.config.CV_NSPLITS, oof_logloss, oof_auc
            )

            logging.getLogger().info(fold_end_msg)
            send_to_telegram(fold_end_msg)

            oof_loglosses.append(oof_logloss)
            oof_aucs.append(oof_auc)

            oof[te_ix] = oof_kfold
            model_name = f'{self.experiment_id}_fold_{i}'

            self.trained_models[model_name] = model
        self._generate_cv_results(oof, oof_loglosses, oof_aucs)

    def crossval_inference(self, X_test, **kwargs):
        """Inference pipeline powered by cross validation. CV fold models
        are required to be already trained.
        """
        if self.trained_models:
            models = self.trained_models.values()
            predict_df = pd.read_csv(self.config.SAMPLE_SUBMISSION)
            preds = np.zeros(predict_df[self.config.LIST_CLASSES].shape)
            for model in models:
                predictions = self.predict(X_test, model, **kwargs)
                preds += predictions

            preds /= len(models)
            predict_df[self.config.LIST_CLASSES] = preds
            self._generate_predictions_file(predict_df)

    def fit_predict(self, X_train=None, X_test=None, y_train=None):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1,
            random_state=self.config.CV_RANDOM_STATE
        )
        if self.config.FOLD_VAL_SPLIT is True:
            model = self.fit(X_train=X_tr, X_val=X_val, y_train=y_tr, y_val=y_val)
        else:
            model = self.fit(X_train=X_train, y_train=y_train)
        predict_df = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        preds = self.predict(X_test, model=model)
        predict_df[self.config.LIST_CLASSES] = preds
        self._generate_predictions_file(predict_df)

    def _generate_predictions_file(self, predict_df):
        predict_filename = f'{self.experiment_id}_predict.csv'
        predict_file_path = os.path.join(self.config.CV_OUTPUT_DIR, f'inference/{predict_filename}')
        predict_df.to_csv(predict_file_path, index=False)
        logging.getLogger().info(f'MODEL: {self.experiment_id}\nINFERENCE SAVED: {predict_file_path}.')

    def _generate_cv_results(self, oof, oof_loglosses, oof_aucs):
        average_logloss = np.sum(oof_loglosses) / len(oof_loglosses)
        average_auc = np.sum(oof_aucs) / len(oof_aucs)

        cv_status_msg = '*** CV COMPLETED ***\n\nMODEL{}\nAVG_LOGLOSS: {:.5f}, AVG_AUC:{:.5f}'.format(
            self.experiment_id, average_logloss, average_auc)

        logging.getLogger().info(cv_status_msg)
        send_to_telegram(cv_status_msg)

        summary_filename = f'{self.experiment_id}_summary.txt'
        oof_filename = f'{self.experiment_id}.csv'
        summary_file_path = os.path.join(self.config.CV_OUTPUT_DIR, summary_filename)
        oof_filename = os.path.join(self.config.CV_OUTPUT_DIR, oof_filename)

        summary_model_info = self.config.display_info()
        summary_result_metric = "Average logloss: {:.5f}, average OOF ROC AUC: {:.5f}".format(average_logloss, average_auc)
        summary = '{}\n\n{}'.format(summary_result_metric, summary_model_info)

        oof_file = pd.read_csv(self.config.TRAIN_DATA_FILE)
        oof_file[self.config.LIST_CLASSES] = oof
        oof_file.to_csv(oof_filename, index=False)

        with open(summary_file_path, 'w') as f:
            f.write(summary)

        logging.getLogger().info(f'MODEL: {self.experiment_id}:\nCV OK. SUMMARY: {summary_file_path}')

    def get_roc_auc_scorer(self, validation_data, interval=1):
        return metrics.RocAucEvaluation(validation_data=validation_data, interval=interval)
