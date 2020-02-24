#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader_v2 import data_loader_v2 # 자체적으로 만든 data loader version 2.0 ([데이콘 15회 대회] 데이터 설명 및 데이터 불러오기 영상 참조)
import joblib # 모델을 저장하고 불러오는 역할


# In[2]:


train_folder = 'data/train/'
test_folder = 'data/test/'
train_label_path = 'data/train_label.csv'


# In[3]:


train_list = os.listdir(train_folder)
test_list = os.listdir(test_folder)
train_label = pd.read_csv(train_label_path, index_col=0)


# In[4]:


def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    
    return combined_df


# In[ ]:


train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, event_time=10, nrows=60)


# In[8]:


X_train = train.drop(['label'], axis=1)
y_train = train['label']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_reg_x = scaler.fit_transform(X_train)


from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)

X_train_pca = pca.fit_transform(train_reg_x)


# In[ ]:


from sklearn import svm


model = svm.SVC(kernel='rbf',probability=True)
model.fit(X_train_pca , y_train)
# joblib.dump(model, 'model.pkl')

test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=10, nrows=60)
pred = model.predict_proba(test)


# In[ ]:


submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'

submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('submission_svm.csv', index=True) #제출 파일 만들기

# logloss값이 randomforest에 비해 매우 좋지 않음.



