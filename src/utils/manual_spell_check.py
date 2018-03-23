import pandas as pd
import re

def replacing (col):
    
    """"Raplace values according to the manual dict"""
    
    for s, new in zip(manual_dict['old'].values, manual_dict['new'].values):
        col = re.sub(r"\b%s\b" % s , new, col)
        
    return col

manual_dict = pd.read_excel('../../data/external/manual_spell_check_v2.xlsx')
print (manual_dict.columns)
train_test = pd.read_json('../../data/interim/train_test_no_dup.json')
train_test['comment_text_manual'] = train_test['comment_text_no_dup'].apply(replacing)
train_test['manual_check'] = (train_test['comment_text_manual']== train_test['comment_text_no_dup']).astype(int)
train_test.to_json('../../data/interim/train_test_manual.json')
