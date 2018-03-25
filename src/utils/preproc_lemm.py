import pandas as pd
from gensim.models import word2vec
import artm
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk import stem
from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import xgboost
from sklearn.linear_model import LogisticRegression
import datetime
pd.options.display.max_colwidth=200
import scipy
import time
import spacy

import telegram

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    # chat_id = -1001111732295
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(15)

def No_with_word(token_text):

    """Concating NO with words"""

    tmp=''
    for i,word in enumerate(token_text):
        if word=='no':
            tmp+=("_".join(token_text[i:i+2]))
            tmp+= ' '
        else:
            if token_text[i-1]!=u'no':
                tmp+=word
                tmp+=' '
    return tmp


def wrk_words_wt_no(sent):

    """Making lemmatisation for Russian and English text"""

    words=word_tokenize(basic_cleaning2(sent).lower())
    words = ' '.join(words)
    # print (words)
    doc = nlp(words)
    # arr = [token.lemma_ for token in doc]
    arr = [w.lemma_ if w.lemma_ != '-PRON-' else w.lower_ for w in doc]

    words1=No_with_word(arr)
    return words1


def basic_cleaning2(string):

    """Cleaning text from numbers and punctuation"""

    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = re.sub(' +', ' ', string)
    return string

# def make_bow(col):
#
#     """Make bow from train_test"""
#
#     binVectorizer = CountVectorizer(binary=True, ngram_range=(1, 1), min_df = 2, max_df=20000, max_features=4000)
# #     binVectorizer = CountVectorizer(binary=True, ngram_range=(1, 1), min_df = 2)
#
#     counts = binVectorizer.fit_transform(np.array(collection))
#     bow_df = pd.DataFrame(counts.toarray(), columns=binVectorizer.get_feature_names())
#     to_include = []
#     for i in bow_df.columns:
#         if bow_df[i].sum()>=2:
#             to_include.append(i)
#     bow_df = bow_df[to_include]
# #     bow_df = scipy.sparse.csr_matrix(bow_df.values)
#     return bow_df




def review_to_wordlist(review):

    """Convert collection to wordlist"""

    words = review.lower().split()
    words = [w for w in words]
    return(words)


# %%time
# train = pd.read_csv('../input/toxic/train.csv', nrows=2000)
# test = pd.read_csv('../input/toxic/test.csv', nrows=2000)

train_test = pd.read_json('../../data/interim/train_test_lang.json')
# train = pd.read_csv('../input/toxic/train.csv')
# test = pd.read_csv('../input/toxic/test.csv')
# cols = train.columns
# sum_of_all = []
# for ind in range(train.shape[0]):
#     temp = train.iloc[ind]
#     sum_of_all.append(temp[cols[2:]].sum())
# train['all'] = sum_of_all
# cols = train.columns
# summing=0
# for i in cols[2:]:
#     if i != 'all':
#         summing+=train[i].sum()
# hot_df = train[train['all']!=0]
# cols = train.columns[2:]
# for i in cols:
#     test[i]=0
# train['train_test'] = 1
# test['train_test'] = 0
# train_test = pd.concat([train, test])
# train_test = train_test.reset_index(drop=True)
# text = train_test['comment_text'][0]
# stemmer=PorterStemmer()
# morph = MorphAnalyzer()
# stop = stopwords.words('english')
nlp = spacy.load('en')
train_test['comment_text_proc'] = train_test['comment_text_corrected'].apply(wrk_words_wt_no)
# train_test = train_test.rename(columns={'comment_text_proc':0})
train_test.to_json('../../data/interim/train_test_lemmed.json')

# send_to_telegram('готово')
