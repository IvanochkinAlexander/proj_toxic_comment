import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import regex as re
import string
import gc
import string
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import wordbatch
from wordbatch.extractors import WordBag

from models.ftrl.config import FTRLConfig
from models.generic_dataset import GenericDataset


class Dataset(GenericDataset):
    def __init__(self, config=FTRLConfig(), **kwargs):
        super().__init__(config, **kwargs)
        train, test = self.data
        #######
        train = train[['comment_text', 'id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
        test = test[['comment_text', 'id']]
        self.X_train, self.X_test, self.y_train = self.preprocess(train, test)

    def preprocess(self, train, test):
        print('filling na')

        train['comment_text'].fillna('', inplace=True)
        test['comment_text'].fillna('', inplace=True)
        print('ok')
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        cont_patterns = [
                (b'US', b'United States'),
                (b'IT', b'Information Technology'),
                (b'(W|w)on\'t', b'will not'),
                (b'(C|c)an\'t', b'can not'),
                (b'(I|i)\'m', b'i am'),
                (b'(A|a)in\'t', b'is not'),
                (b'(\w+)\'ll', b'\g<1> will'),
                (b'(\w+)n\'t', b'\g<1> not'),
                (b'(\w+)\'ve', b'\g<1> have'),
                (b'(\w+)\'s', b'\g<1> is'),
                (b'(\w+)\'re', b'\g<1> are'),
                (b'(\w+)\'d', b'\g<1> would'),
            ]
        patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

        def prepare_for_char_n_gram(text):
            """ Simple text clean up process"""
            # 1. Go to lower case (only good for english)
            # Go to bytes_strings as I had issues removing all \n in r""
            clean = bytes(text.lower(), encoding="utf-8")
            # 2. Drop \n and  \t
            clean = clean.replace(b"\n", b" ")
            clean = clean.replace(b"\t", b" ")
            clean = clean.replace(b"\b", b" ")
            clean = clean.replace(b"\r", b" ")
            # 3. Replace english contractions
            for (pattern, repl) in patterns:
                clean = re.sub(pattern, repl, clean)
            # 4. Drop puntuation
            # I could have used regex package with regex.sub(b"\p{P}", " ")
            exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
            clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
            # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
            clean = re.sub(b"\d+", b" ", clean)
            # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
            clean = re.sub(b'\s+', b' ', clean)
            # Remove ending space if any
            clean = re.sub(b'\s+$', b'', clean)
            # 7. Now replace words by words surrounded by # signs
            # e.g. my name is bond would become #my# #name# #is# #bond#
            # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
            clean = re.sub(b" ", b"# #", clean)  # Replace space
            clean = b"#" + clean + b"#"  # add leading and trailing #

            return str(clean, 'utf-8')

        @contextmanager
        def timer(name):
            """
            Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
            in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
            https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
            """
            t0 = time.time()
            yield
            print(f'[{name}] done in {time.time() - t0:.0f} s')

        def count_regexp_occ(regexp="", text=None):
            """ Simple way to get the number of occurence of a regex"""
            if type(text) is str:
                return len(re.findall(regexp, text))
            else:
                return 0

        def get_indicators_and_clean_comments(df):
            """
            Check all sorts of content as it may help find toxic comment
            Though I'm not sure all of them improve scores
            """
            # Count number of \n
            df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
            # Get length in words and characters
            df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
            df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
            # Check number of upper case, if you're angry you may write in upper case
            df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
            # Number of F words - f..k contains folk, fork,
            df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
            # Number of S word
            df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
            # Number of D words
            df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
            # Number of occurence of You, insulting someone usually needs someone called : you
            df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))

            df["nb_bitch"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wbitch\W", x))
            df["nb_crap"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wcrap\W", x))
            df["nb_jew"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wjew[s]\W", x))

            # Just to check you really refered to my mother ;-)
            df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
            # Just checking for toxic 19th century vocabulary
            df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
            # Some Sentences start with a <:> so it may help
            df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
            # Check for time stamp
            df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
            # Check for dates 18:44, 8 December 2010
            df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
            # Check for date short 8 December 2010
            df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
            # Check for http links
            df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
            # check for mail
            df["has_mail"] = df["comment_text"].apply(
                lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
            )
            # Looking for words surrounded by == word == or """" word """"
            df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
            df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

            # Now clean comments
            df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

            # Get the new length in words and characters
            df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
            df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
            # Number of different characters used in a comment
            # Using the f word only will reduce the number of letters required in the comment
            df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
            df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
                lambda x: 1 + min(99, len(x)))

        def char_analyzer(text):
            """
            This is used to split strings in small lots
            I saw this in an article (I can't find the link anymore)
            so <talk> and <talking> would have <Tal> <alk> in common
            """
            tokens = text.split()
            return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

        def clean_csr(csr_trn, csr_sub, min_df):
            trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
            sub_min = {x for x in np.where(csr_sub.getnnz(axis=0) >= min_df)[0]}
            mask= [x for x in trn_min if x in sub_min]
            return csr_trn[:, mask], csr_sub[:, mask]

        def get_numerical_features(trn, sub):
            """
            As @bangda suggested FM_FTRL either needs to scaled output or dummies
            So here we go for dummies
            """
            ohe = OneHotEncoder()
            full_csr = ohe.fit_transform(np.vstack((trn.values, sub.values)))
            csr_trn = full_csr[:trn.shape[0]]
            csr_sub = full_csr[trn.shape[0]:]
            del full_csr
            gc.collect()

            # Now remove features that don't have enough samples either in train or test
            return clean_csr(csr_trn, csr_sub, 3)

        class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        with timer("Performing basic NLP"):
            for df in [train, test]:
                get_indicators_and_clean_comments(df)

        train_text = train['clean_comment'].fillna("")
        test_text = test['clean_comment'].fillna("")
        all_text = pd.concat([train_text, test_text])

        with timer("Creating numerical features"):
            num_features = [f_ for f_ in train.columns
                            if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars", 'has_ip_address'] + class_names]

            # skl = MinMaxScaler()
            # skl.fit(pd.concat([train[num_features], test[num_features]], axis=0))
            # train[num_features] = skl.transform(train[num_features])
            # test[num_features] = skl.transform(test[num_features])
            # FM_FTRL requires categorical data
            train.fillna("", inplace=True)
            test.fillna("", inplace=True)
            test['comment_text'] = test['comment_text'].astype(str)
            test['clean_comment'] = test['clean_comment'].astype(str)
            for f in num_features:
                all_cut = pd.cut(pd.concat([train[f], test[f]], axis=0), bins=20, labels=False, retbins=False)
                train[f] = all_cut.values[:train.shape[0]]
                test[f] = all_cut.values[train.shape[0]:]

            train_num_features, test_num_features = get_numerical_features(train[num_features], test[num_features])
            # train_num_features = csr_matrix(train[num_features])
            # test_num_features = csr_matrix(test[num_features])

        with timer("Tfidf on word"):
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                tokenizer=lambda x: re.findall(r'[^\p{P}\W]+', x),
                analyzer='word',
                token_pattern=None,
                ngram_range=(1, 2),  # was (1 , 2)
                max_features=self.config.TFIDF_WORD_FEATURES)
            word_vectorizer.fit(all_text)
            train_word_features = word_vectorizer.transform(train_text)
            test_word_features = word_vectorizer.transform(test_text)

        with timer("Tfidf on char n_gram"):
            char_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                tokenizer=char_analyzer,
                analyzer='word',
                ngram_range=(1, 3),
                max_features=self.config.TFIDF_CHAR_FEATURES)
            char_vectorizer.fit(all_text)
            train_char_features = char_vectorizer.transform(train_text)
            test_char_features = char_vectorizer.transform(test_text)

        with timer("Staking matrices"):
            train_features = hstack(
                [
                    train_char_features,
                    train_word_features,
                    train_num_features
                ]
            ).tocsr()
            del train_word_features, train_num_features, train_char_features
            gc.collect()

            test_features = hstack(
                [
                    test_char_features,
                    test_word_features,
                    test_num_features
                ]
            ).tocsr()
            del test_word_features, test_num_features, test_char_features
            gc.collect()
        return train_features, test_features, train[self.config.LIST_CLASSES].values
