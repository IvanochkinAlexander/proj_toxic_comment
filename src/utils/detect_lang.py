import pandas as pd
from textblob import TextBlob
import telegram
import time

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(15)

def detect_lang(col):

    """Detects two languages in topic"""

    try:
        shape = int(len(col)/2)
        col1 = col[:shape]
        col2 = col[shape:]
        return TextBlob(col1).detect_language(),TextBlob(col2).detect_language()
    except:
        return 'Unkmown'

### executing script
train_test=pd.read_json('../../data/interim/train_test_corrected.json')
# train_test = train_test.rename(columns={'0':0})
train_test = train_test.reset_index(drop=True)
# train_test= train_test[:100]
train_test['lang'] =train_test['comment_text_corrected'].apply(detect_lang)

def split_col_1 (col):
    try:
        col1 = col[0]
        return col1
    except:
        return 'unknown'

train_test['lang_1'] = train_test['lang'].apply(split_col_1)

def split_col_2 (col):
    try:
        col1 = col[1]
        return col1
    except:
        return 'unknown'

train_test['lang_2'] = train_test['lang'].apply(split_col_2)
train_test['one_lang'] = (train_test['lang_1']==train_test['lang_2']).astype(int)



# send_to_telegram('язык готов')
train_test.to_json('../../data/interim/train_test_lang.json')
