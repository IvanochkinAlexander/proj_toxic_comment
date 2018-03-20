from models.generic_config import GenericConfig


class LogisticRegressionConfig(GenericConfig):
    AUGMENTATION = False
    TFIDF_WORD_FEATURES = 15000
    TFIDF_CHAR_FEATURES = 30000
