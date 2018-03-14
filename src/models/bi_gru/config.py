import os
from models.generic_config import GenericConfig
from collections import OrderedDict


class BiGRUConfig(GenericConfig):
    PROCESSED_TRAIN_TEST = os.path.join(GenericConfig.DATA_DIR,
                                        'interim/train_test_manual_no.json')

    FEATURES_COLUMN = 'comment_text_manual_no'

    CHECKPOINT_PATH = os.path.join(GenericConfig.DATA_DIR, 'last_chpt.h5')
    EMBEDDINGS = {
        'max_features': 25000,
        'embed_size': 300,
        'maxlen': 150
    }

    LEARNING_RATE = 1e-3
    DECAY = 0.0

    DROPOUT = 0.2

    UNITS = OrderedDict({
        'GRU': 128,
        'Conv1D': 64,
        'Dense': 6
    })

    BATCH_SIZE = 32
    EPOCHS = 2

# change to 80k
# https://www.kaggle.com/eashish/bidirectional-lstm-with-convolution
class BiGRU85kConfig(BiGRUConfig):
    EMBEDDINGS = {
        'max_features': 85000,
        'embed_size': 200,
        'maxlen': 250
    }

    LEARNING_RATE = 1e-3
    DECAY = 0.0

    DROPOUT = 0.2

    # 0.1 was a default one, 0.3 is said to work better (see below)
    GRU_DROPOUT = 0.3

    # 140 units, 0.3 dropout
    UNITS = OrderedDict({
        'GRU': 140,
        'Conv1D': 64,
        'Dense': 6
    })

    BATCH_SIZE = 32

    # 5 epochs may work too
    EPOCHS = 5
