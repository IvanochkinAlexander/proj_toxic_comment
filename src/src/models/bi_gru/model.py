from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import BiGRUConfig
from dataset import Dataset

class BiGRU(GenericModel):
    MODEL_NAME = 'bigru_conv_erlstop_1'

    def __init__(self):
        super()
        self.config = BiGRUConfig
        self.dataset = Dataset()

    def build_model(self):
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

    def fit(self, model, X_train, X_val, y_train, y_val):
        check_point = ModelCheckpoint(self.config.CHECKPOINT_PATH, monitor="val_loss",
                                      verbose=1, save_best_only=True, mode='min')
        ra_val = self.get_roc_auc_scorer(validation_data=(X_val, y_val), interval=1)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

        model.fit(X_train, y_train, batch_size=self.config.BATCH_SIZE,
                  epochs=self.config.EPOCHS,
                  validation_data=(X_val, y_val), verbose=1,
                  callbacks=[ra_val, check_point, early_stop])
        model.load_weights(self.config.CHECKPOINT_PATH)
        return model
