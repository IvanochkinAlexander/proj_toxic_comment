from models.generic_config import ProcessedModelConfig

class FTRLConfig(ProcessedModelConfig):
    TFIDF_WORD_FEATURES = 300000
    TFIDF_CHAR_FEATURES = 60000

    ITERATIONS = 3
