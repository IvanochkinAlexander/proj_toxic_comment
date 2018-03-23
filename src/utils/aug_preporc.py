import pandas as import pd

"""
Preprocessing augmented text according to method of Pavel Ostyakov
"""

def compare_bad (col, col_old):

    """Comparing text from original and augmented"""

    add_text = ''
    for i in col_old:
        for w in i:
            if w in col:
                add_text+=w
                add_text+=' '
    return add_text.strip()

def read_augs (path):

    """Reading all files"""

    all_lang = pd.DataFrame()
    for i in os.listdir(path):
        temp = pd.read_csv(path + i)
        temp['lang'] = i.split('_')[1].split('.csv')[0]
        all_lang = pd.concat([all_lang, temp], axis=0)
        return read_augs(path)

def preproc_aug (all_lang, new_train_test_add, temp_json):

    """Saving preproc file"""

    train_test_filt = new_train_test_add[['id']+list(target_cols)]
    train_test_filt['train_test']=0
    train_test_filt.loc[train_test_filt['toxic']!='', 'train_test'] = 1
    train_test_filt['is_toxic_text'] = 0
    for i in target_cols:
        train_test_filt.loc[train_test_filt[i]==1, 'is_toxic_text'] = 1
    train_test_markers = train_test_filt[['id', 'train_test', 'is_toxic_text']].reset_index(drop=True)
    all_lang = pd.merge(all_lang, train_test_markers, on='id', how='left')
    all_lang_train = all_lang[all_lang['train_test']==1]
    all_lang_train = all_lang_train[all_lang_train['is_toxic_text']==1]
    all_lang_train.to_json('../projects/proj_toxic_comment/data/interim/train_test_aug_lemmed.json.json')
    train_test_aug_lemmed = pd.read_json('../projects/proj_toxic_comment/data/interim/train_test_aug_lemmed.json')
    train_test_aug_lemmed['splited_comment_text_proc'] = train_test_aug_lemmed['comment_text_proc'].apply(lambda x :[i for i in x.split(' ')])
    bad_words = pd.read_csv('../projects/proj_toxic_comment/data/external/compiled_bad_words.txt', sep='/n', header=None)
    train_test_aug_lemmed['bad_in_new'] = train_test_aug_lemmed['splited_comment_text_proc'].apply(lambda x: search_bad_general(x, bad_words))
    train_test_aug_lemmed = train_test_aug_lemmed.rename(columns={'comment_text':'comment_text_aug'})
    original = new_train_test_add[['id', 'comment_text_manual_no_extra_2']]
    train_test_aug_lemmed = pd.merge(train_test_aug_lemmed, original, on='id', how='left')
    train_test_aug_lemmed['splited_comment_extra_2'] = train_test_aug_lemmed['comment_text_manual_no_extra_2'].apply(lambda x: [i for i in x.split(' ')])
    train_test_aug_lemmed['bad_in_old'] = train_test_aug_lemmed['splited_comment_extra_2'].apply(lambda x: search_bad_general(x, bad_words))
    col_old = train_test_aug_lemmed['bad_in_old'].values
    train_test_aug_lemmed['diff'] = train_test_aug_lemmed['bad_in_new'].apply(lambda x: compare_bad(x, col_old))
    all_texts = []
    for i, k in zip (train_test_aug_lemmed['bad_in_old'].values, train_test_aug_lemmed['bad_in_new'].values):
        add_text = ''
        for w in i:
            if w not in k:
                add_text+=w
                add_text+=' '
        all_texts.append(add_text)
    train_test_aug_lemmed['diff'] = all_texts
    train_test_aug_lemmed['comment_text_proc_extra'] =train_test_aug_lemmed['diff'] +' ' + train_test_aug_lemmed['comment_text_proc']
    train_test_aug_lemmed['comment_text_proc_extra'] =train_test_aug_lemmed['comment_text_proc_extra'].apply(lambda x : x.strip())
    train_test_all = new_train_test_add[['id']+list(target_cols)]
    train_all = train_test_all[train_test_all['toxic']!='']
    train_test_aug_lemmed = pd.merge(train_test_aug_lemmed, train_all, on='id', how='left')
    del train_test_aug_lemmed['comment_text_manual_no_extra_2']
    train_test_aug_lemmed = train_test_aug_lemmed.rename(columns={'comment_text_proc_extra':'comment_text_manual_no_extra_2'})
    new_to_merge = train_test_aug_lemmed[['id']+['comment_text_manual_no_extra_2']+list(target_cols)]
    old_to_merge = new_train_test_add[['id']+['comment_text_manual_no_extra_2']+list(target_cols)]
    train_test_extended_plus_30k = pd.concat([new_to_merge, old_to_merge]).reset_index(drop=True)
    train_test_extended_plus_30k['comment_text_manual_no_extra_2'] = train_test_extended_plus_30k['comment_text_manual_no_extra_2'].apply(lambda x : x.replace('_', ' '))
    train_test_extended_plus_30k.to_json('../projects/proj_toxic_comment/data/interim/train_test_extended_plus_30k_reversed.json')

temp_json = pd.read_json('../toxic/tools/train_test_manual_no_extra_2.json')
temp_json = temp_json.rename(columns={'comment_text_manual_no_extra_2': 'comment_text'})
path = '../toxic/tools/extended_data/'
all_lang =read_augs (path)
