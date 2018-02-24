import numpy as np
import pandas as pd
import telegram
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(5)

def basic_cleaning2(string):

    """Cleaning text from numbers and punctuation"""

    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = re.sub(' +', ' ', string)
    return string

train_test = pd.read_json('../../data/interim/train_test_ascii.json')
# print train_test.columns
# print train_test.shape
### spliting again to use external cleaning script
train = train_test[train_test['train_test']==1]
test = train_test[train_test['train_test']==0]
# train = pd.read_csv('../../../input/toxic/train.csv')[:200]
# test = pd.read_csv('../../../input/toxic/test.csv')[:200]
# submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

# PREPROCESSING PART
repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["ascii_text"].tolist()
lte = test["ascii_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www' or len(j)>50:
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www' or len(j)>50:
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data
# print("crap removed")
trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()

for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', ' ', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', ' ', tete[i])
train["comment_text_cleaned"] = trate
test["comment_text_cleaned"] = tete

print (train.columns)
print (test.columns)
# print (train.columns)
# target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# for col in target_cols:
    # test[col] = ''
# test['train_test'] = 0
# train['train_test'] = 1
train_test = pd.concat([train, test]).reset_index(drop=True)
print (train_test.columns)
print (train_test.shape)
train_test['comment_text_cleaned'] = train_test['comment_text_cleaned'].apply(lambda x : basic_cleaning2(x))
train_test.to_json('../../data/interim/train_test_cleaned.json')
# print (test.columns)
# print('only alphabets')
# send_to_telegram('replaced')
