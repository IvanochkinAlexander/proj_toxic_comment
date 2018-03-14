import logging
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models.generic_dataset import GenericDataset

import pandas as pd
import numpy as np


class Dataset(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        train, test = self.data
        train_test, tokenizer = self.generate_sequences(train, test)
        self.X_train, self.X_test, self.y_train = train_test
        self.embedding_matrix = self.get_embedding_matrix(tokenizer)

    def generate_sequences(self, train, test):
        list_sentences_train = train["comment_text"].fillna("_na_").values
        list_sentences_test = test["comment_text"].fillna("_na_").values

        tokenizer = Tokenizer(num_words=self.config.EMBEDDINGS['max_features'])
        tokenizer.fit_on_texts(list(list_sentences_train))

        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

        X_train = pad_sequences(list_tokenized_train, maxlen=self.config.EMBEDDINGS['maxlen'])
        y_train = train[self.config.LIST_CLASSES].values

        X_test = pad_sequences(list_tokenized_test, maxlen=self.config.EMBEDDINGS['maxlen'])
        logging.getLogger().info('Dataset: Sequences generation completed.')
        return (X_train, X_test, y_train), tokenizer

    def _get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def _get_embedding_index(self):
        return dict(self._get_coefs(*o.strip().split(" "))
                    for o in open(self.config.EMBEDDING_FILE, encoding='utf-8'))

    def get_embedding_matrix(self, tokenizer):
        embedding_index = self._get_embedding_index()
        word_index = tokenizer.word_index
        nb_words = min(self.config.EMBEDDINGS['max_features'], len(word_index))
        embedding_matrix = np.zeros((nb_words, self.config.EMBEDDINGS['embed_size']))
        for word, i in word_index.items():
            if i >= self.config.EMBEDDINGS['max_features']: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        logging.getLogger().info('Dataset: Embedding matrix calculated.')
        return embedding_matrix
