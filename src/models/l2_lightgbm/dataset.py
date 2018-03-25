import numpy as np
import pandas as pd
import os
from models.generic_dataset import GenericDataset


class Dataset(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_names = []
        self.X_train = self.prepare_oofs()
        print('OOFs prepared.')
        self.y_train = self.prepare_labels()
        print('Labels prepared.')
        self.X_test = self.prepare_inferences()
        print('Inference prepared.')

    def prepare_labels(self):
        print('Preparing labels....')
        data_train = pd.read_csv(self.config.TRAIN_DATA_FILE)
        data_train.sort_values(by='id', inplace=True)
        return data_train[self.config.LIST_CLASSES].astype(np.float).values

    def prepare_oofs(self):
        oofs = []
        for oof_file in sorted(os.listdir(self.config.L1_OOF_DIR)):
            full_path = os.path.join(self.config.L1_OOF_DIR, oof_file)
            model_name = oof_file.split('__')[1]
            model_df = pd.read_csv(full_path)
            model_df.sort_values(by='id', inplace=True)
            oofs.append((model_name, model_df))
        data_oofs = np.hstack([np.array(oof_model[self.config.LIST_CLASSES])
                               for _, oof_model in oofs])
        self.model_names = [model_name for model_name, _ in oofs]
        return data_oofs

    def prepare_inferences(self):
        inferences = []
        for inf_file in sorted(os.listdir(self.config.L1_INFERENCE_DIR)):
            full_path = os.path.join(self.config.L1_INFERENCE_DIR, inf_file)
            model_name = inf_file.split('__')[1]
            model_df = pd.read_csv(full_path)
            model_df.sort_values(by='id', inplace=True)
            inferences.append((model_name, model_df))
        if sorted([model_name for model_name, _ in inferences]) == sorted(self.model_names):
            data_infs = np.hstack([np.array(inf_model[self.config.LIST_CLASSES])
                                         for _, inf_model in inferences])
            return data_infs
        raise RuntimeError('Models in OOF and INF do not match, terminating.')

    def add_engineered_features(self):
        FEATURES_JSON_PATH = os.path.join(self.config.DATA_DIR,
                                          'processed/l3_added_features/features_v2.json')
        FEATURES = ['caps_vs_length', 'em_vs_length', 'num_words',
                    'punkt_vs_length', 'qm_vs_length', 'smilies_vs_words',
                    'symb_vs_length', 'total_length', 'words_vs_bad_words',
                    'words_vs_unique', 'not_en']
        features_df = pd.read_json(FEATURES_JSON_PATH)
        train_df = pd.read_csv(self.config.TRAIN_DATA_FILE)
        test_df = pd.read_csv(self.config.TEST_DATA_FILE)
        train_df = pd.merge(train_df, features_df, how='left', on='id').reset_index()
        train_df.sort_values(by='id', inplace=True)
        test_df = pd.merge(test_df, features_df, how='left', on='id').reset_index()
        test_df.sort_values(by='id', inplace=True)
        self.X_train = np.hstack([self.X_train, np.array(train_df[FEATURES])])
        self.X_test = np.hstack([self.X_test, np.array(test_df[FEATURES])])
        print('Added engineered features')
