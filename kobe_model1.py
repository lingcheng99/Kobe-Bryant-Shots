import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, train_test_split,cross_val_score
from sklearn import metrics

#input df with both train and test
def base_feature(data):
    #at home or not
    data['home?'] = data['matchup'].apply(lambda x: x.split('@')[0].strip()=='LAL')
    #2-point or 3-point shot
    data['shot_type?'] = data['shot_type']=='2PT Field Goal'
    #playoff or not
    data['playoffs?'] = data['playoffs']==1
    return data

#encode categorical variable into sorted integer, and plotting
def sort_encode(df, field):
    ct = pd.crosstab(df.shot_made_flag, df[field]).apply(lambda x:x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    df[new_field] = df[field].map(new_map)
    get_acc(df, new_field)


#plot one col in df against shot_made_flag percentage
def get_acc(df, col):
    ct = pd.crosstab(df.shot_made_flag, df[col]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(col)
    plt.ylabel('% shots made')
    #plt.savefig(against + '_vs_accuracy.png')

#base modeling with randomforest
def test_accuracy(data): #input data with cols_use and 'shot_made_flag'
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=7) #specify tree and depth
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,scoring='accuracy', cv=10)

# another test function with roc_auc as metrics instead of accuracy
def test_auc(data): #input data with cols_use and 'shot_made_flag'
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=7) #specify tree and depth
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,\
                           scoring='roc_auc', cv=10)



if __name__=='__main__':
    data = pd.read_csv('data.csv')
    #create some base features
    data = base_feature(data)

    #enumerate action_type first
    action_map = {action: i for i, action in enumerate(data.action_type.unique())}
    data['action_type_enumerated'] = data.action_type.map(action_map)
    #add column 'action_type_enumerated_sort_enumerated'
    sort_encode(data, 'action_type_enumerated')

    #base modeling
    cols_use=['action_type_enumerated_sort_enumerated',
         'playoffs?','home?', 'shot_type?', 'shot_distance','shot_made_flag']
    df = data[cols_use3]
    df = df.dropna()
    print test_auc(df).mean()
    print test_accuracy(df).mean()

    #from data_exploration_1, add loc_y and abs_loc_x
    data['abs_loc_x'] = data['loc_x'].apply(lambda x: abs(x))
    cols_use1=['action_type_enumerated_sort_enumerated','abs_loc_x','loc_y',
         'playoffs?','home?', 'shot_type?', 'shot_distance','shot_made_flag']
    df = data[cols_use1]
    df = df.dropna()
    print test_auc(df).mean()
    print test_accuracy(df).mean()
