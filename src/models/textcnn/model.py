from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from .config import TextCNNConfig

# Preprocessing is the same as for GRU/LSTM
from models.bi_gru.dataset import Dataset

from models.bi_gru.model import BiGRU


class TextCNN(BiGRU):
    MODEL_NAME = 'textcnn'

    def __init__(self, config=TextCNNConfig(), **kwargs):
        super().__init__(config, **kwargs)

    def _build_model(self):
        inp = Input(shape=(self.config.EMBEDDINGS['maxlen'], ))
        x = Embedding(self.config.EMBEDDINGS['max_features'], self.config.EMBEDDINGS['embed_size'], weights=[self.dataset.embedding_matrix])(inp)
        x = SpatialDropout1D(self.config.SPATIAL_DROPOUT)(x)
        x = Reshape((self.config.EMBEDDINGS['maxlen'], self.config.EMBEDDINGS['embed_size'], 1))(x)

        conv_0 = Conv2D(self.config.NUM_FILTERS,
                        kernel_size=(self.config.FILTER_SIZES[0],
                                     self.config.EMBEDDINGS['embed_size']),
                        kernel_initializer='normal', activation='elu')(x)
        conv_1 = Conv2D(self.config.NUM_FILTERS,
                        kernel_size=(self.config.FILTER_SIZES[1],
                                     self.config.EMBEDDINGS['embed_size']),
                        kernel_initializer='normal', activation='elu')(x)
        conv_2 = Conv2D(self.config.NUM_FILTERS,
                        kernel_size=(self.config.FILTER_SIZES[2],
                                     self.config.EMBEDDINGS['embed_size']),
                        kernel_initializer='normal', activation='elu')(x)
        conv_3 = Conv2D(self.config.NUM_FILTERS,
                        kernel_size=(self.config.FILTER_SIZES[3],
                                     self.config.EMBEDDINGS['embed_size']),
                        kernel_initializer='normal', activation='elu')(x)

        maxpool_0 = MaxPool2D(pool_size=(self.config.EMBEDDINGS['maxlen'] - self.config.FILTER_SIZES[0] + 1, 1))(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.config.EMBEDDINGS['maxlen'] - self.config.FILTER_SIZES[1] + 1, 1))(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.config.EMBEDDINGS['maxlen'] - self.config.FILTER_SIZES[2] + 1, 1))(conv_2)
        maxpool_3 = MaxPool2D(pool_size=(self.config.EMBEDDINGS['maxlen'] - self.config.FILTER_SIZES[3] + 1, 1))(conv_3)

        z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = Flatten()(z)
        z = Dropout(self.config.DENSE_DROPOUT)(z)

        outp = Dense(6, activation="sigmoid")(z)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
