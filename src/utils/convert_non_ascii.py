import pandas as pd
from textblob import TextBlob
import telegram
import time
import unidecode

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(5)

def unidecode_text (col):
    try:
        return unidecode.unidecode(col)
    except:
        return str(col) + ' too low symbols'

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

### executing script
# train_test = pd.read_csv('../../data/train_test_lang_splited.csv', nrows=1000)

train = pd.read_csv('../../../input/toxic/train.csv')
test = pd.read_csv('../../../input/toxic/test.csv')

target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for col in target_cols:
    test[col] = ''
test['train_test'] = 0
train['train_test'] = 1
train_test = pd.concat([train, test]).reset_index(drop=True)


# train_test= train_test[:100]
# train_test = train_test.rename(columns={'0':0})
# train_test = train_test.reset_index(drop=True)

train_test['ascii_text'] = train_test['comment_text'].apply(unidecode_text)
train_test['check is ascii']  = train_test['ascii_text'].apply(is_ascii)
# train_test['lang'] =train_test[0].apply(detect_lang)

# train_test.to_csv('../../../output/train_test_ascii.csv')

train_test.to_json('../../data/interim/train_test_ascii.json')
# send_to_telegram('ascii готов')
