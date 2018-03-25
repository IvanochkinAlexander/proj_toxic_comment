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

"""
Making lemmatisation of augmented text
"""


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

def review_to_wordlist(review):

    """Convert collection to wordlist"""

    words = review.lower().split()
    words = [w for w in words]
    return(words)


train_test = pd.read_json('../../data/interim/train_test_aug.json')
nlp = spacy.load('en')
train_test['comment_text_proc'] = train_test['comment_text'].apply(wrk_words_wt_no)
train_test.to_json('../../data/interim/train_test_aug_lemmed_all.json')
