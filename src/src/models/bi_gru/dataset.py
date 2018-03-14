from config import BiGRUConfig
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class Dataset:
    def __init__(self):
        self.tokenizer = None
        train, test = self.prepare_data()
        self.X_train, self.X_test, self.y_train = self.generate_sequences(train, test)
        self.embedding_matrix = self.get_embedding_matrix()

    def prepare_data(self):
        train = pd.read_csv(BiGRUConfig.TRAIN_DATA_FILE)
        test = pd.read_csv(BiGRUConfig.TEST_DATA_FILE)
        processed = pd.read_json(BiGRUConfig.PROCESSED_TRAIN_TEST)

        train = train[[col for col in train.columns if col != 'comment_text']]
        test = test[[col for col in test.columns if col != 'comment_text']]

        processed = processed[['id', 'comment_text_manual_no']]
        processed.rename(columns={'comment_text_manual_no': 'comment_text'},
                         inplace=True)
        train = pd.merge(train, processed, how='left', on='id')
        test = pd.merge(test, processed, how='left', on='id')

        return train, test

    def generate_sequences(self, train, test):
        list_sentences_train = train["comment_text"].fillna("_na_").values
        list_sentences_test = test["comment_text"].fillna("_na_").values

        self.tokenizer = Tokenizer(num_words=BiGRUConfig.EMBEDDINGS['max_features'])
        self.tokenizer.fit_on_texts(list(list_sentences_train))

        list_tokenized_train = self.tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)

        X_train = pad_sequences(list_tokenized_train, maxlen=BiGRUConfig.EMBEDDINGS['maxlen'])
        y_train = train[BiGRUConfig.LIST_CLASSES].values

        X_test = pad_sequences(list_tokenized_test, maxlen=BiGRUConfig.EMBEDDINGS['maxlen'])
        return X_train, X_test, y_train

    def _get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def _get_embedding_index(self):
        return dict(self._get_coefs(*o.strip().split(" "))
                    for o in open(BiGRUConfig.EMBEDDING_FILE, encoding='utf-8'))

    def get_embedding_matrix(self, tokenizer):
        embedding_index = self._get_embedding_index()
        word_index = tokenizer.word_index
        nb_words = min(BiGRUConfig.EMBEDDINGS['max_features'], len(word_index))
        embedding_matrix = np.zeros((nb_words, BiGRUConfig.EMBEDDINGS['embed_size']))
        for word, i in word_index.items():
            if i >= BiGRUConfig.EMBEDDINGS['max_features']: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
