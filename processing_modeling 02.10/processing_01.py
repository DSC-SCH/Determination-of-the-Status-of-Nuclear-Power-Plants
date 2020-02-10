#!/usr/bin/env python
# coding: utf-8

# In[105]:


import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader_v2 import data_loader_v2 # 자체적으로 만든 data loader version 2.0 ([데이콘 15회 대회] 데이터 설명 및 데이터 불러오기 영상 참조)

from sklearn.ensemble import RandomForestClassifier


# In[106]:


train_folder = 'data/train/'
test_folder = 'data/test/'
train_label_path = 'data/train_label.csv'


# In[107]:


train_list = os.listdir(train_folder)
test_list = os.listdir(test_folder)
train_label = pd.read_csv(train_label_path, index_col=0)


# In[108]:


def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df


# In[110]:


train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, event_time=10, nrows=60)


# In[109]:


test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=10, nrows=60)


# In[90]:


X_train = train.drop(['label'], axis=1)


# In[91]:


y_train = train['label']


# In[ ]:


print(train.shape)
print(test.shape)


# In[93]:


X_train = train.drop(['label'], axis=1)
y_train = train['label']

total = pd.concat([X_train,test])
train = total.fillna(0) # 모든 NA값을 0으로 대체

name = total.columns

for i in name:
    if total[i].dtypes=='object':
        del total[i]
        
train_reg_x = total[0:41400]
test_reg_x = total[41400:]        


# In[96]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
total_reg_x = scaler.fit_transform(total)


# In[97]:


total = total.reset_index()


# In[98]:


del total['index']


# In[100]:


test_reg_x.shape


# In[101]:


train_reg_x = train_reg_x.fillna(0)
test_reg_x = test_reg_x.fillna(0)


# In[102]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1)
model.fit(train_reg_x, y_train)
# joblib.dump(model, 'model.pkl')


# In[103]:


pred = model.predict_proba(test_reg_x)


# In[112]:


test.index


# In[113]:


submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('submission.csv', index=True) #제


# In[114]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[ ]:


model = XGBClassifier()
model.fit(train_reg_x, y_train)
pred = model.predict_proba(test_reg_x)


# In[ ]:


submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('submission_xgboost.csv', index=True) #제


# In[ ]:




