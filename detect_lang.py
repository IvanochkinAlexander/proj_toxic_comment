import pandas as import pd
from textblob import textblob

train_test=pd.read_json('../input/toxic/train_test.json')


train_test = train_test.rename(columns={'0':0})

def detect_lang(col):
    try:
        col1 = col[:len(col)/2]
        col2 = col[len(col/2):]
        return TextBlob(col1).detect_language(),TextBlob(col2).detect_language()
    except:
        return 'Unkmown'

train_test = train_test.reset_index(drop=True)

train_test= train_test[:1000]

%%time
train_test['lang'] =train_test[0].apply(detect_lang)
train_test.to_csv('../output/train_test_lang.csv')
