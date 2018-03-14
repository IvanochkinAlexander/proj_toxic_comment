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
from sklearn.metrics import accuracy_score

import telegram

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    # bot = telegram.Bot(token='523412387:AAHhEckKtZiCoSG6Pd3ZGtp4-JbL06I8H2E')
    # 523412387:AAHhEckKtZiCoSG6Pd3ZGtp4-JbL06I8H2E
    # chat_id = -1001111732295
    chat_id = 169719023

    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(5)

def No_with_word(token_text):

    """Concating NO with words"""

    tmp=''
    for i,word in enumerate(token_text):
        if word==u'не':
            tmp+=("_".join(token_text[i:i+2]))
            tmp+= ' '
        else:
            if token_text[i-1]!=u'не':
                tmp+=word
                tmp+=' '
    return tmp


def wrk_words_wt_no(sent):

    """Making lemmatisation for Russian and English text"""

    words=word_tokenize(basic_cleaning2(sent).lower())
    lemmatizer = WordNetLemmatizer()
    try:
        arr=[]
        for i in range(len(words)):
            arr.append(morph.parse(words[i])[0].normal_form)
        arr2 = []
        for i in arr:
            arr2.append(lemmatizer.lemmatize(i, pos='v'))
        arr3 = []
        for i in arr2:
            arr3.append(lemmatizer.lemmatize(i, pos='n'))
        arr4 = []
        for i in arr3:
            arr4.append(lemmatizer.lemmatize(i, pos='a'))
        words1=[w for w in arr4 if w not in stop]
#         words1=[w for w in arr4 if w not in english_stops]
        words1=No_with_word(words1)
        return words1
    except TypeError:
        pass

def basic_cleaning2(string):

    """Cleaning text from numbers and punctuation"""

    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = re.sub(' +', ' ', string)
    return string






train_test=pd.read_json('../input/toxic/train_test.json')
# train_test_small_first  = train_test[:300]
# train_test_small_last  = train_test.tail(300)
# train_test = pd.concat([train_test_small_first, train_test_small_last])
train_test = train_test.rename(columns={'0':0})
collection = train_test[0]
train_test = train_test.reset_index(drop=True)
train_index = train_test[train_test['train_test']==1].index
test_index = train_test[train_test['train_test']==0].index

bow_df, test_bow = make_bow(pd.DataFrame(collection).iloc[train_index][0], pd.DataFrame(collection).iloc[test_index][0])
w2v, model = make_w2v(collection)
send_to_telegram('загружен файл')
w2v_tfidf_df =  make_w2v_tfidf(collection)
send_to_telegram('препроцессинг сделан')
