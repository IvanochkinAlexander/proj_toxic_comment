import logging
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from models.generic_model import GenericModel
from .config import BiGRUConfig, BiGRU85kConfig
from .dataset import Dataset


class BiGRU(GenericModel):
    MODEL_NAME = 'bigru_conv1d'

    def __init__(self, config=BiGRUConfig()):
        super().__init__()
        self.config = config
        self.dataset = Dataset(config=config)

    def fit(self, X_train=None, X_val=None, y_train=None, y_val=None):
        model = self._build_model()
        if X_val is not None:
            check_point = ModelCheckpoint(self.config.CHECKPOINT_PATH, monitor="val_loss",
                                          verbose=1, save_best_only=True, mode='min')
            ra_val = self.get_roc_auc_scorer(validation_data=(X_val, y_val), interval=1)
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

            model.fit(X_train, y_train, batch_size=self.config.BATCH_SIZE,
                      epochs=self.config.EPOCHS,
                      validation_data=(X_val, y_val), verbose=1,
                      callbacks=[ra_val, check_point, early_stop])
            model.load_weights(self.config.CHECKPOINT_PATH)
        else:
            model.fit(X_train, y_train, batch_size=self.config.BATCH_SIZE,
                      epochs=self.config.EPOCHS, verbose=1)
        return model

    def _build_model(self):
        inp = Input(shape=(self.config.EMBEDDINGS['maxlen'],))
        x = Embedding(self.config.EMBEDDINGS['max_features'],
                      self.config.EMBEDDINGS['embed_size'],
                      weights=[self.dataset.embedding_matrix], trainable=False)(inp)
        x = SpatialDropout1D(self.config.DROPOUT)(x)

        x = Bidirectional(GRU(self.config.UNITS['GRU'], return_sequences=True))(x)
        x = Conv1D(self.config.UNITS['Conv1D'], kernel_size=2, padding="valid",
                   kernel_initializer="he_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        x = Dense(self.config.UNITS['Dense'], activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=self.config.LEARNING_RATE,
                                     decay=self.config.DECAY),
                      metrics=["accuracy"])
        return model


# https://www.kaggle.com/eashish/bidirectional-lstm-with-convolution
class BiGRU85k(BiGRU):
    MODEL_NAME = 'bigru_85k'

    def __init__(self, config=BiGRU85kConfig()):
        super().__init__()
        self.config = config
        self.dataset = Dataset(config=config)

    def _build_model(self):
        inp = Input(shape=(self.config.EMBEDDINGS['maxlen'], ))
        x = Embedding(self.config.EMBEDDINGS['max_features'],
                      self.config.EMBEDDINGS['embed_size'],
                      weights=[self.dataset.embedding_matrix],
                      trainable=False)(inp)
        x = SpatialDropout1D(self.config.DROPOUT)(x)

        x = Bidirectional(GRU(self.config.UNITS['GRU'],
                              return_sequences=True,
                              dropout=self.config.GRU_DROPOUT,
                              recurrent_dropout=self.config.GRU_DROPOUT))(x)
        x = Conv1D(self.config.UNITS['Conv1D'],
                   kernel_size=3,
                   padding="valid", kernel_initializer="glorot_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        x = Dense(self.config.UNITS['Dense'], activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer=Nadam(lr=self.config.LEARNING_RATE),
                      metrics=['accuracy'])
        return model
