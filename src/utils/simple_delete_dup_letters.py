import pandas as pd
import re

import telegram
import time

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='523412387:AAHhEckKtZiCoSG6Pd3ZGtp4-JbL06I8H2E')
    # chat_id = -1001371737931
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(5)


train_test= pd.read_json('../../data/interim/simple_train_test_cleaned_v2.json')
# train_test = train_test[:1000]
print ('loaded')
# train_test = train_test
train_test['comment_text_splited'] = train_test['comment_text_cleaned'].apply(lambda x : x.split(' '))
correct_words = pd.read_json('../../data/external/double_words.json')
correct_words = correct_words['words'].unique()

def delete_dup_letters (col):
    try:
        all_words = ''
        for w_ind in range(len(col)):
            if col[w_ind].lower() not in correct_words:
                all_words+=re.sub(r'([a-z])\1+', r'\1', col[w_ind])
                all_words+=' '
            else:
                all_words+=col[w_ind]
                all_words+=' '
        return all_words
    except:
        return col

train_test['comment_text_no_dup'] = train_test['comment_text_splited'].apply(delete_dup_letters)
train_test.to_json('../../data/interim/simple_train_test_no_dup.json')
send_to_telegram('дубликаты удалены')
