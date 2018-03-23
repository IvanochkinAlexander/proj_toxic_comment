import pandas as import pd

"""
This script corrects data after spell checking,
searches wrong spelled words like 'f.u.ckk',
addes them to the top of string
"""

def generate_check_bad (bad_words):

    """Generates list of wrong spelled words like b_i_t_c_h"""

    bad_words = pd.DataFrame(bad_words)
    bad_words = bad_words.drop_duplicates(subset=0, keep='first')
    symbols = ['.', '_', '*', ' ', '|', '/']

    new_words=[]
    true_words = []
    for sym in symbols:
        for i in bad_words[0].unique():
            new_word = ''
            for num in range(len(i)):
                new_word+= i[num]
                new_word+=sym
            new_words.append(new_word)
            true_words.append(i)

    for i in bad_words[0]:
        mask = '*'*len(i)
        new_word = i[0]+mask[1:]
        new_words.append(new_word)
        true_words.append(i)
        new_word_2 = i[:1]+mask[1:]
        new_words.append(new_word_2)
        true_words.append(i)
        new_word_3 = i[:2]+mask[2:]
        new_words.append(new_word_3)
        true_words.append(i)
    new_words = new_words+['f*ck', 'f**k', 'd*ck', 'b*tch', 'facking', 'c*nt', 'pr*ck', 's*ck']
    true_words = true_words+['fuck', 'fuck', 'dick', 'bitch', 'fucking', 'cunt', 'prick', 'suck']

    check_bad = pd.DataFrame(new_words, true_words).reset_index()
    check_bad.columns = ('actual', 'modified')
    check_bad = check_bad.drop_duplicates(subset='modified', keep='first')
    check_bad['new'] = check_bad['actual'].apply(lambda x : '*' in x).astype(int)
    check_bad = check_bad[check_bad['new']!=1]

    return check_bad

def search_bad (col, check_bad):

    """Searching bad words"""

    all_str = ''
    try:

        for i, k in zip(check_bad['modified'], check_bad['actual']):
            if i in col.split(' '):
                all_str+=k
                all_str+=' '
        return all_str.strip()

    except:

        return 0

def replace_wrong_bad (train_test, manual_3, check_bad):

    """Adds correct words to the top"""

    train_test['ascii_text'] = train_test['ascii_text'].apply(lambda x: x.lower())
    train_test['ascii_text_check_2'] = train_test['ascii_text'].apply(lambda x: search_bad(x, check_bad))
    train_test['comment_text_manual_no_check_2'] = train_test['comment_text_manual_no'].apply(lambda x: search_bad(x, check_bad))
    train_test['ascii_text_check_3'] = train_test['ascii_text'].apply(lambda x: search_bad(x, manual_3))
    new_train_test = train_test
    check_cols = ['ascii_text_check_2', 'ascii_text_check_3']
    for i in check_cols:
        new_train_test['len_' + i] = new_train_test[i].apply(lambda x : len(x))
    for i in check_cols:
        print (i, new_train_test[new_train_test['len_'+i]>0].shape[0])
    check_cols = ['ascii_text_check_2', 'ascii_text_check_3', 'comment_text_manual_no']
    for i in check_cols:
        new_train_test['splited_' + i] = new_train_test[i].apply(lambda x : [i for i in x.split(' ')])
    all_vals = []
    for i, k in zip(new_train_test['splited_ascii_text_check_2'].values, new_train_test['splited_comment_text_manual_no'].values):
        new_val = ''
        for val in i:
            if not val in k:
                new_val+=val
                new_val+=' '
        all_vals.append(new_val.strip())
    new_val_1 = pd.DataFrame(all_vals)
    all_vals = []
    for i, k in zip(new_train_test['splited_ascii_text_check_3'].values, new_train_test['comment_text_manual_no'].values):
        new_val = ''
        for val in i:
            if not val in k:
                new_val+=val
                new_val+=' '
        all_vals.append(new_val.strip())
    new_val_2 = pd.DataFrame(all_vals)
    new_vals = pd.concat([new_val_1, new_val_2], axis=1)
    new_vals.columns = ('add_bad_words_1', 'add_bad_words_2')
    new_train_test_add = pd.concat([new_train_test.reset_index(drop=True), new_vals.reset_index(drop=True)],axis=1)
    for i in ['add_bad_words_1', 'add_bad_words_2']:
        new_train_test_add['len_'+i] = new_train_test_add[i].apply(lambda x: len(x))
    new_train_test_add['comment_text_manual_no_extra']=new_train_test_add['comment_text_manual_no']
    for i in ['add_bad_words_1', 'add_bad_words_2']:
        new_train_test_add.loc[new_train_test_add['len_'+i] >2, 'comment_text_manual_no_extra'] =  new_train_test_add[i] + ' ' +new_train_test_add['comment_text_manual']
    new_train_test_add['comment_text_manual_no_extra'] = new_train_test_add['comment_text_manual_no_extra'].apply(lambda x: x.strip())
    new_train_test_add['comment_text_manual_no_extra_check_2'] = new_train_test_add['comment_text_manual_no_extra'].apply(lambda x: search_bad_splited(x, check_bad))
    new_train_test_add['len_comment_text_manual_no_extra_check_2'] = new_train_test_add['comment_text_manual_no_extra_check_2'].apply(lambda x: len(x))
    new_train_test_add['comment_text_manual_no_extra_2'] = new_train_test_add['comment_text_manual_no_extra']
    new_train_test_add.loc[new_train_test_add['len_comment_text_manual_no_extra_check_2'] >2,'comment_text_manual_no_extra_2'] = new_train_test_add['comment_text_manual_no_extra_check_2'] + ' '+ new_train_test_add['comment_text_manual_no_extra_2']
    ids_to_check = new_train_test_add.loc[new_train_test_add['len_comment_text_manual_no_extra_check_2'] >2]['id'].unique()

    return new_train_test_add


def search_bad_splited (col, check_bad):
    all_str = ''
    try:
        for i, k in zip(check_bad['modified'], check_bad['actual']):
            if i in col:
                all_str+=k
                all_str+=' '
        return all_str.strip()
    except:
        return 0




bad_words =[
'fuck',
'shit',
'motherfucker',
'cock',
'cunt',
'dick',
'bitch',
'faggot',
'nigger',
'tits',
'asshole',
'cum',
'clit',
'dildo',
'vagina',
'twat',
'ass',
'damn',
'bastard',
'boobs',
'anal',
'ass fucker',
'bullshit',
'cunillingus',
'dumbass',
'slut',
'suck',
'wank',
'breast',
'butt',
'crap',
'dyke',
'erection',
'homo',
'pussy',
'vulva',
'wtf',
'anus',
'fag',
'fucker',
'gay',
'jerk',
'lesbian',
'nazi',
'penis',
'piss',
'porn',
'prostitute',
'rape',
'shity'
]

train_test = pd.read_json('../projects/proj_toxic_comment/data/interim/train_test_manual_no.json')
manual_3 = pd.read_csv('../projects/proj_toxic_comment/data/external/manual_check_v3.csv', sep=';')
check_bad = generate_check_bad(bad_words)
new_train_test_add = replace_wrong_bad (train_test, manual_3, check_bad)
new_train_test_add.to_json('../projects/proj_toxic_comment/data/interim/train_test_manual_no_extra_reversed.json')
