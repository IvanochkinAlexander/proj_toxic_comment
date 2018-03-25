import os


class GenericConfig:
    MODEL_PREFIX = 'model'

    DATA_DIR = os.path.join('..', 'data')
    CV_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed/cross_validation')
    LOG_DIR = os.path.join(DATA_DIR, 'processed/cross_validation/logs')

    SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'raw/sample_submission.csv')

    EMBEDDING_FILE = os.path.join(DATA_DIR, 'external/glove.840B.300d.txt')
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'raw/train.csv')
    TEST_DATA_FILE = os.path.join(DATA_DIR, 'raw/test.csv')

    PROCESSED_TRAIN_TEST = os.path.join(DATA_DIR, 'interim/train_test_manual_no_extra_2.json')
    FEATURES_COLUMN = 'comment_text_manual_no_extra_2'

    CV_NSPLITS = 5
    CV_RANDOM_STATE = 12
    FOLD_VAL_SPLIT = False

    AUGMENTATION = False

    LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def display_info(self):
        config_info = "\nConfiguration:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config_info += ("\n{:30} {}".format(a, getattr(self, a)))
        config_info += "\n"
        return config_info


class ProcessedModelConfig(GenericConfig):
    PROCESSED_TRAIN_TEST = os.path.join(GenericConfig.DATA_DIR,
                                        'interim/train_test_manual_no_extra.json')
    FEATURES_COLUMN = 'comment_text_manual_no_extra'
