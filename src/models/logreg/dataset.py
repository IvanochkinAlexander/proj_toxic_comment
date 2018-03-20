from models.generic_dataset import GenericDataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.send_to_telegram import send_to_telegram


class Dataset(GenericDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        train, test = self.data
        self.X_train, self.X_test, self.y_train = self.vectorize(train, test)

    def vectorize(self, train, test):
        train_text = train['comment_text']
        test_text = test['comment_text']
        all_text = pd.concat([train_text, test_text])
        send_to_telegram('LR: Vec started.')

        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            max_features=self.config.TFIDF_WORD_FEATURES)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)
        send_to_telegram('LR: Word vec completed.')

        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(2, 6),
            max_features=self.config.TFIDF_CHAR_FEATURES)
        char_vectorizer.fit(all_text)
        send_to_telegram('LR: Char vec completed.')

        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

        train_features = np.hstack([train_char_features, train_word_features])
        test_features = np.hstack([test_char_features, test_word_features])
        send_to_telegram('LR: hstack completed.')
        return train_features, test_features, train[self.config.LIST_CLASSES].values
