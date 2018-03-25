import pandas as pd

def read_oof (path_oof):

    """Reading public oofs"""

    all_oofs = pd.DataFrame()
    for i in os.listdir(path_oof):
        temp = pd.read_csv(path+i)
        temp['name'] = i.split('.csv')[0]
        all_oofs = pd.concat([all_oofs, temp],axis=0)
    all_inf.reset_index(drop=True).to_json('../projects/proj_toxic_comment/data/interim/all_inf.json')

def read_inf (path_inf):

    """Reading public inferences and making mean"""

    all_inf = pd.DataFrame()
    for i in os.listdir(path_inf):
        print (i)
        temp = pd.read_csv(path+i)
        target_cols = temp.columns[1:]
        target_cols = target_cols[:6]
        for col in target_cols:
            temp[col] = temp.groupby('id')[col].transform('mean')
        temp = temp.drop_duplicates(subset='id', keep='first')
        temp['name'] = i.split('.csv')[0]
        all_inf = pd.concat([all_inf, temp],axis=0)
    all_oofs.reset_index(drop=True).to_json('../projects/proj_toxic_comment/data/interim/all_oofs.json')

path_oof =  '../projects/input/toxic/oof_neptune/oof/'
path_inf =  '../projects/input/toxic/oof_neptune/inference/'
read_oof (path_oof)
read_inf (path_inf)
