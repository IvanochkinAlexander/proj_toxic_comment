import os
from models.bi_gru.config import BiGRUConfig


class TextCNNConfig(BiGRUConfig):
    PROCESSED_TRAIN_TEST = os.path.join(BiGRUConfig.DATA_DIR,
                                        'interim/train_test_manual_no_extra.json')

    EMBEDDING_FILE = os.path.join(BiGRUConfig.DATA_DIR, 'external/glove.twitter.27B.200d.txt')
    FEATURES_COLUMN = 'comment_text_manual_no_extra'

    CHECKPOINT_PATH = os.path.join(BiGRUConfig.DATA_DIR, 'last_chpt.h5')
    EMBEDDINGS = {
        'max_features': 30000,
        'embed_size': 200,
        'maxlen': 400
    }

    FILTER_SIZES = [1, 2, 3, 5]
    NUM_FILTERS = 32

    LEARNING_RATE = 1e-3

    SPATIAL_DROPOUT = 0.4
    DENSE_DROPOUT = 0.1

    BATCH_SIZE = 256
    EPOCHS = 3
