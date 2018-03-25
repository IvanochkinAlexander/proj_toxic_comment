import pandas as pd

def blend_it (sub_3, sub_73, new_early):

    """Average of 3 different predictions"""

    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']
    new_blend = (sub_73[target_cols]+sub_3[target_cols]+new_early[target_cols])/3
    new_blend = pd.concat([sub_3[['id']], new_blend],axis=1)
    new_blend.to_csv('../projects/input/toxic/subs_true/sub_4_temp.csv', index=False)

sub_3 = pd.read_csv('../projects/input/toxic/subs_true/sub_3_v2.csv')
sub_73 = pd.read_csv('../projects/input/toxic/subs_true/0.9873_stacked_58_public_67_public_60.csv')
new_early = pd.read_csv('../projects/input/toxic/subs_true/L2_COMBO_10_MAX_DOP2_FIN__l2_catboost_20180320_225519__predict.csv')

blend_it(sub_3, sub_73, new_early)
