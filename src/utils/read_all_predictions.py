import os
import pandas as pd
import seaborn as sns
import matplotlib as plt
%pylab inline

def read_all_preds (path):

    """Reading all the predictions for test to analyse it"""

    names = []
    all_df = pd.DataFrame()
    for i in os.listdir(path):
        if i[-3:]=='csv':
            names.append(i.split('.csv')[0])
            temp = pd.read_csv(path+i)
            temp['sub_name'] = i.split('.csv')[0]
            all_df= pd.concat([all_df, temp],axis=0)
    return all_df

def save_table (all_df, new_train_test_add, name):

    """Saving predictions in proper format to look through
    It is neccesary because of imbalansed dataset"""

    texts_df = new_train_test_add[['id', 'comment_text_manual_no_extra_2']]
    target_cols = target_cols[:6]
    pivot_mean = all_df.pivot_table(target_cols, 'sub_name', aggfunc='mean').reset_index().sort_values('toxic')
    for i in target_cols:
        all_df[i+'_round'] = all_df[i].apply(lambda x : x > 0.5).astype(int)
    target_cols_pred_round = target_cols.map(lambda x: x+'_round')
    pivot_count = all_df.pivot_table(target_cols_pred_round, 'sub_name', aggfunc='sum').reset_index()
    pivot_count.set_index('sub_name').style.background_gradient(cmap='RdYlGn',axis=1)
    to_merge = new_train_test_add[['id', 'comment_text', 'comment_text_manual_no_extra_2']]
    two_to_comapre = all_df[all_df['sub_name'].isin([name])]
    all_concated = pd.DataFrame()
    for i in two_to_comapre['sub_name'].unique():
        print (i)
        temp = two_to_comapre[two_to_comapre['sub_name']==i]
        for col in temp.columns:
            temp=temp.rename(columns={col:col+'_{}'.format(i[:10])})
        all_concated = pd.concat([all_concated, temp], axis=1)
    to_merge['len_str'] = to_merge['comment_text_manual_no_extra_2'].apply(lambda x : len(x))
    to_merge['len_words'] = to_merge['comment_text_manual_no_extra_2'].apply(lambda x : len(x.split(' ')))
    new_train_test_add['comment_text_manual_no_extra_2_splited'] = new_train_test_add['comment_text_manual_no_extra_2'].apply(lambda x: x.split(' '))
    new_train_test_add['bad_check'] =new_train_test_add['comment_text_manual_no_extra_2_splited'].apply(lambda x: [i for i in x if i in bad_words[0].values])
    compared = pd.merge(all_concated.reset_index(drop=True), to_merge.reset_index(drop=True), left_on='id_L2_COMBO_1', right_on='id', how='left')
    compared = compared.sort_values(['len_words', 'len_str'], ascending=False)
    del compared['comment_text']
    compared.to_excel('../projects/proj_toxic_comment/data/processed/to_check_data_l2_all_combo_catboost_early.xlsx')

path = '../projects/input/toxic/subs_true/'
all_df = read_all_preds (path)
new_train_test_add=pd.read_json('../projects/proj_toxic_comment/data/interim/train_test_manual_no_extra_new_reversed.json')
name = 'L2_COMBO_10_MAX_DOP2__l2_catboost_20180320_220228__predict'
save_table(all_df, new_train_test_add, name)
