import jamspell
import pandas as pd
import time
import telegram
import unidecode

print ('start')
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('../../data/external/en.bin')
check = corrector.LoadLangModel('../../data/external/en.bin')
print (check)
print ('loaded model')
train_test = pd.read_json('../../data/interim/train_test_cleaned.json')
# train_test= train_test[:200]

# def send_to_telegram(text):
#
#     """Send appropriate links to telegram channel"""
#
#     bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
#     chat_id = 169719023
#     bot.send_message(chat_id=chat_id, text=text)
#     time.sleep(15)


# print (train_test.columns)
def correct_text (col):
    try:
        print (col)
        return corrector.FixFragment(col)

    except:
        return col

def unidecode_text (col):
    try:
        return unidecode.unidecode(col)
    except:
        return str(col) + ' too low symbols'

# train_test['comment_text_corrected'] = train_test['comment_text_cleaned'].apply(lambda x : corrector.FixFragment(x))
train_test['comment_text_cleaned'] = train_test['comment_text_cleaned'].apply(unidecode_text)
train_test['comment_text_corrected'] = train_test['comment_text_cleaned'].apply(correct_text)

print ('corrected')
train_test.to_json('../../data/interim/train_test_corrected.json')
print ('complete')

# send_to_telegram('spell_checker finished')
