import os


class GenericConfig:
    DATA_DIR = os.path.join('../..', 'data')
    CV_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed/cross_validation')

    SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'raw/sample_submission.csv')

    EMBEDDING_FILE = os.path.join(DATA_DIR, 'external/glove.840B.300d.txt')
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'raw/train.csv')
    TEST_DATA_FILE = os.path.join(DATA_DIR, 'raw/test.csv')

    CV_NSPLITS = 5
    CV_RANDOM_STATE = 2

    LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    @staticmethod
    def display_info():
        raise NotImplementedError
