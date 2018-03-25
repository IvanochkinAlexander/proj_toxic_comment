import pandas as pd
import re


train_test = pd.read_json('../../data/interim/train_test_manual.json')

def replacing (col):

    col = re.sub('_', ' ', col)

    return col

train_test['comment_text_manual_no'] = train_test['comment_text_manual'].apply(replacing)

train_test.to_json('../../data/interim/train_test_manual_no.json')
