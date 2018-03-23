import pandas as pd

new_train_test_add = new_train_test_add = pd.read_json('../projects/proj_toxic_comment/data/interim/train_test_manual_no_extra_reversed.json')
bad_words = pd.read_csv('../projects/proj_toxic_comment/data/external/compiled_bad_words.txt', sep='/n', header=None)

def search_bad (col):

    """For bad words count"""

    all_str = ''
    try:
        for k in bad_words[0]:
            if k in col.split(' '):
                all_str+=k
                all_str+=' '
        return all_str.strip()
    except:
        return 0

def generate_features (new_train_test_add , bad_words):

    """Some features for L2- stacking"""

    df = new_train_test_add
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
    df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df.loc[df['ascii_text_check_2'] !='', 'all_bad'] = df['ascii_text_check_2']
    df.loc[df['ascii_text_check_3'] !='', 'all_bad'] = df['ascii_text_check_3']
    df['all_bad_2'] = df['comment_text'].apply(search_bad)
    df.loc[df['all_bad_2']!='', 'all_bad'] = df['all_bad']
    test_preproc = pd.read_csv('../projects/proj_toxic_comment/data/external/test_preprocessed.csv')
    train_preproc = pd.read_csv('../projects/proj_toxic_comment/data/external/train_preprocessed.csv')
    external_train_test = pd.concat([test_preproc, train_preproc])[['id', 'comment_text']]
    external_train_test.drop_duplicates(subset='id', keep='first').shape
    external_train_test = external_train_test.rename(columns = {'comment_text':'comment_text_external'})
    df_merged = pd.merge(df, external_train_test, on ='id', how='left')
    df_merged['all_bad_external'] = df_merged['comment_text_external'].apply(search_bad)
    df_merged.loc[df_merged['all_bad'].isnull(), 'all_bad'] = df_merged['all_bad_2']
    df_merged.loc[df_merged['all_bad_external']=='', 'all_bad_external'] = df_merged['all_bad']
    new_train_test_add['not_en'] = 0
    new_train_test_add.loc[~new_train_test_add['lang_1'].isin(['en', 'de', 'es', 'fr', 'it']) & ~new_train_test_add['lang_2'].isin(['en', 'de', 'es', 'fr', 'it']), 'not_en']=1
    new_train_test_add.loc[new_train_test_add['lang']=='Unkmown', 'not_en'] = 0
    new_train_test_add[new_train_test_add['not_en']==1]
    features_new = pd.merge(features_check, new_train_test_add[['id', 'not_en']], on='id', how='left')
    features_new.to_json('../projects/proj_toxic_comment/data/interim/features_v2.json')
