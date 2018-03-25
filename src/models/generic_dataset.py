import logging
import pandas as pd


class GenericDataset:
    def __init__(self, config, use_json=True, **kwargs):
        logging.getLogger().info('Dataset: Generation started.')
        self.use_neptune = False
        self.config = config
        if use_json is True:
            self.data = self.prepare_data()
        else:
            train = pd.read_csv(self.config.TRAIN_DATA_FILE)
            test = pd.read_csv(self.config.TEST_DATA_FILE)
            self.data = train, test

    def prepare_data(self):
        train = pd.read_csv(self.config.TRAIN_DATA_FILE)
        test = pd.read_csv(self.config.TEST_DATA_FILE)
        processed = pd.read_json(self.config.PROCESSED_TRAIN_TEST)

        features_column = self.config.FEATURES_COLUMN

        train = train[[col for col in train.columns if col != 'comment_text']]
        test = test[[col for col in test.columns if col != 'comment_text']]

        processed = processed[['id', features_column]]
        processed.rename(columns={features_column: 'comment_text'},
                         inplace=True)
        train = pd.merge(train, processed, how='left', on='id')
        test = pd.merge(test, processed, how='left', on='id')
        logging.getLogger().info('Dataset: Data loaded.')
        return train, test
