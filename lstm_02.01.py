#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random

from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(11)

from sklearn.model_selection import train_test_split

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


def data_loader_all_v2(func, files, folder='', train_label=None, event_time=random.randrange(0,16), nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    
    return combined_df


# In[ ]:


train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, event_time=random.randrange(0,16), nrows=60)



# In[ ]:


test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=random.randrange(0, 16), nrows=60)


# In[6]:

# 시간 모델을 고려한 lstm autoencoder 사용
# lstm 모델을 위한 데이터 정제
# lstm 모델은 3D array로 만들어야함
# sample: 데이터 수
# lookback: lstm model에서 과거 어디까지 볼 것 인가
# features: 인풋으로 사용할 개수
X_input = train.drop(['label'], axis=1).values
y_input = train['label'].values


# In[ ]:

n_features = X_input.shape[1]

# lookback을 추가하여 3D array로 변환
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y


lookback = 1 # Equivalent to 10 min of past data.
# Temporalize the data

X, y = temporalize(X = X_input, y = y_input, lookback = lookback)


# In[ ]:

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2

X_train, y_train = np.array(X), np.array(y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)


# In[ ]:

X_train_y0 = X_train[y_train==0]
X_train_y1 = X_train[y_train==1]

X_valid_y0 = X_valid[y_valid==0]
X_valid_y1 = X_valid[y_valid==1]


# In[ ]:

lookback = 1

X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)

X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)

input_test = test.values
X_test = np.array(input_test)
X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

# In[ ]:

# 3D array를 2D로 변환하는 함수 생성
def flatten(X):

    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)


# 데이터 표준화
# 2D 데이터에서 표준화를 하면 3D test데이터에서 위치 정보를 잃어버리기 때문에 유효한 검사 불가
def scale(X, scaler):

    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
        
    return X


# In[ ]:


scaler = StandardScaler().fit(flatten(X_train_y0))


X_train_y0_scaled = scale(X_train_y0, scaler)
X_train_y1_scaled = scale(X_train_y1, scaler)
X_train_scaled = scale(X_train, scaler)


X_valid_scaled = scale(X_valid, scaler)
X_valid_y0_scaled = scale(X_valid_y0, scaler)

X_test_scaled = scale(X_test, scaler)


# In[ ]:


timesteps =  X_train_y0_scaled.shape[1] 
n_features =  X_train_y0_scaled.shape[2] 

epochs = 100
batch_size = 64
lr = 0.0001


# In[ ]:

lstm_autoencoder = Sequential()

# Encoder
# softmax함수를 이용하여 class값을 예측하고자함
# 하지만 softmax함수는 수식 안에 e의 지수를 포함하고 있어 지수가 커질수록 매우 큰 폭으로 증가하기 때문에 overflow가 발생하기 쉬움.

#def new_softmax(a) : 
#    c = np.max(a) # 최댓값
#    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취함(overflow 방지)
#    sum_exp_a = np.sum(exp_a)
#    y = exp_a / sum_exp_a
#    return y

lstm_autoencoder.add(LSTM(32, activation='softmax', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(4, activation='softmax', return_sequences=False, dropout = 0.5))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(4, activation=new_softmax, return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation=new_softmax, return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()


# In[ ]:
# 방향성과 stepsize 모두를 고려하기 위해 여러 경사하강법 중 Adam 사용
# - RMSprop의 특징인 gradient의 제곱을 지수평균한 값을 사용
# - Momentum의 특징으로 gradient를 제곱하지 않은 값을 사용하여 지수평균을 구하고 수식에 활용함
adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled, 
                                                epochs=epochs, 
                                                batch_size=batch_size, 
                                                validation_data=(X_valid_y0_scaled, X_valid_y0_scaled),
                                                verbose=2).history




# In[ ]:


train_x_predictions = lstm_autoencoder.predict(X_train_scaled)

mse = np.mean(np.power(flatten(X_train_scaled) - flatten(train_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_train.tolist()})

groups = error_df.groupby('True_class')

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")


# In[ ]:


valid_x_predictions = lstm_autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(flatten(X_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_valid.tolist()})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)


# In[ ]:

# 예측 
test_x_predictions = lstm_autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(flatten(X_test_scaled) - flatten(test_x_predictions), 2), axis=1)


# In[ ]:


intermediate_layer = Model(inputs=lstm_autoencoder.inputs, outputs=lstm_autoencoder.layers[1].output)
# time_dist_layer = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[5].output)
intermediate_output = intermediate_layer.predict(X_train_y0_scaled)


# In[ ]:


intermediate_output

