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
train_test=pd.read_json('../input/toxic/train_test.json')
train_test = train_test.rename(columns={'0':0})
train_test = train_test.reset_index(drop=True)
train_test= train_test[:100]
train_test['lang'] =train_test[0].apply(detect_lang)
send_to_telegram('язык готов')
train_test.to_csv('../output/train_test_lang.csv')
