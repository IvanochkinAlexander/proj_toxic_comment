import os
from ..generic_config import GenericConfig


class BiGRUConfig(GenericConfig):

    PROCESSED_TRAIN_TEST = os.path.join(GenericConfig.DATA_DIR,
                                        'interim/train_test_manual_no.json')
    CHECKPOINT_PATH = os.path.join(GenericConfig.DATA_DIR, 'last_chpt.h5')
    EMBEDDINGS = {
        'max_features': 25000,
        'embed_size': 300,
        'maxlen': 150
    }

    LEARNING_RATE = 1e-3
    DECAY = 0.0

    DROPOUT = 0.2

    UNITS = {
        'GRU': 128,
        'Conv1D': 64,
        'Dense': 6
    }

    BATCH_SIZE = 32
    EPOCHS = 4

    @staticmethod
    def display_info(self):
        config_info = "Configuration:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config_info += ("\n{:30} {}".format(a, getattr(self, a)))
        config_info += "\n"
        return config_info
