import pandas as pd
from sklearn.metrics import roc_auc_score

def read_oof (path):

    """Analysing quality of oof predictions"""

    oof_all = pd.DataFrame()

    for filename in os.listdir(path):
        if filename.startswith('L2_ALL_COMBO_CB__l2_catboost_20180320_150415'):
            print (filename)
            oof = pd.read_csv(path+ filename)
            print (oof.shape[0])
            oof = pd.merge(oof, texts_df, on='id', how='left')
            train_true = new_train_test_add[list(target_cols)+['id']]
            for i in target_cols:
                train_true = train_true.rename(columns={i:i+'_true'})
            oof = pd.merge(oof, train_true, on ='id', how='left')
            print (oof.shape)
            oof['sub_name'] = filename.split('.csv')[0]
            oof_all = pd.concat([oof_all, oof])
            scores = []
            for i in target_cols:
                print ('{} class:'.format(i).upper())
                y_true = oof[i+'_true'].astype(int).values
                y_scores = oof[i].values
                scores.append(roc_auc_score(y_true, y_scores))
                print ('ROC AUC Score is {}'.format("%.4f" % roc_auc_score(y_true, y_scores)))
                print ('Total number of samples', oof[oof[i+'_true']==1].shape[0])
                oof[i+'_pred_round'] = oof[i].apply(lambda x: x >=0.5).astype(int)
                print ('Median_value is {}'.format("%.4f" % oof[i].median()))
                print ('Mean_value is {}'.format("%.4f" % oof[i].mean()))
                oof[i+'_error'] = (oof[i+'_pred_round'] != oof[i+'_true']).astype(int)
                oof[i+'_correct'] = (oof[i+'_pred_round'] == oof[i+'_true']).astype(int)
                print ('Correct count {}'.format(oof[oof[i+'_true']==1].shape[0] - oof[oof[i+'_error']==1].shape[0]))
                print ('Error count {}'.format(oof[oof[i+'_error']==1].shape[0]))
                oof[i+'_false_positive'] =  (oof[i+'_pred_round']> oof[i+'_true']).astype(int)
                oof[i+'_false_negative'] =  (oof[i+'_pred_round']< oof[i+'_true']).astype(int)
                print ('False positive {}'.format(oof[oof[i+'_false_positive']==1].shape[0]))
                print ('False negative {}'.format(oof[oof[i+'_false_negative']==1].shape[0]))
                print ('***')
            print ('Mean ROC AUC value is {}'.format("%.4f" % np.mean(scores)))
            print (' ')
            print ('*****')
            print (' ')

path= '../projects/input/toxic/subs_oof/'
new_train_test_add=pd.read_json('../projects/proj_toxic_comment/data/interim/train_test_manual_no_extra_new_reversed.json')
texts_df = new_train_test_add[['id', 'comment_text_manual_no_extra_2']]
read_oof(path)
