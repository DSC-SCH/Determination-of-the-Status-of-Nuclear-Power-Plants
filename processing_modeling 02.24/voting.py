import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader_v3 import data_loader_v3
from sklearn.ensemble import RandomForestClassifier

train_folder = 'data/train/'
test_folder = 'data/test/'
train_label_path = 'data/train_label.csv'

train_list = os.listdir(train_folder)
test_list = os.listdir(test_folder)
train_label = pd.read_csv(train_label_path, index_col=0)

def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df
    
train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, event_time=10, nrows=60)
test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=10, nrows=60)
 
X_train = train.drop(['label'], axis=1)
y_train = train['label']
 
print(train.shape)
print(test.shape)

X_train = train.drop(['label'], axis=1)
y_train = train['label']

total = pd.concat([X_train,test])
total = total.fillna(0) # 모든 NA값을 0으로 대체

name = total.columns

for i in name:
    if total[i].dtypes=='object':
        del total[i]

total[name[0]].dtypes=='object'

train_reg_x = total[0:41350]
test_reg_x = total[41350:] 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
total_reg_x = scaler.fit_transform(total)

total = total.reset_index()

del total['index']

test_reg_x.shape
total.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
# VotingClassifier
from sklearn.ensemble import VotingClassifier

# ensemble 할 model 정의
models = [
    ('ada', AdaBoostClassifier()),
    ('bc', BaggingClassifier()),
    ('etc',ExtraTreesClassifier()),
    ('gbc', GradientBoostingClassifier()),
    ('rfc', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier()),
    ('lgbm', LGBMClassifier()),
    ('dtc', DecisionTreeClassifier()),
    ('lr', LogisticRegressionCV()),
    ('ridge', RidgeClassifier()),
]

# hard vote
hard_vote  = VotingClassifier(models, voting='hard')
hard_vote.fit(train_reg_x, y_train)

# soft vote
soft_vote  = VotingClassifier(models, voting='soft')
soft_vote.fit(train_reg_x, y_train)

pred = hard_vote.predict_proba(test_reg_x)
pred_soft = soft_vote.predict_proba(test_reg_x)

submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('submission_hard.csv', index=True) 

submission = pd.DataFrame(data=pred_soft)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('submission_soft.csv', index=True)
