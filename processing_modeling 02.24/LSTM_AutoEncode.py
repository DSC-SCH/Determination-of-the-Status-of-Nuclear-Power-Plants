# Reference - https://github.com/DSC-SCH/Determination-of-the-Status-of-Nuclear-Power-Plants/issues/6
import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader_2 import data_loader_v2 # 자체적으로 만든 data loader version 2.0 ([데이콘 15회 대회] 데이터 설명 및 데이터 불러오기 영상 참조)

# keras
from keras.models import Sequential
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

import joblib # 모델을 저장하고 불러오는 역할

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

train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, 
                        train_label=train_label, event_time=10, nrows=60)

X_train = np.asarray(train.drop(['label'], axis=1))
y_train = np.asarray(train['label'])

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''

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

timesteps = 5
X, y = temporalize(X = X_train, y = y_train, lookback = timesteps)

# n_features
n_features = len(X_train[0])
X = np.array(X)

x_valid_index = int(len(X) * 0.2)
y_valid_index = int(len(y) * 0.2)


# split train / validation set
X_train = X[:x_valid_index]
X_valid = X[x_valid_index:]
y_train = y[:y_valid_index]
y_valid = y[y_valid_index:]

# reshpae
X_train = X_train.reshape(X_train.shape[0], timesteps, n_features)
X_valid = X_valid.reshape(X_valid.shape[0], timesteps, n_features)


def flatten(X):
    '''
    Flatten a 3D array.
    
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def scale(X, scaler):
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
        
    return X

scaler = StandardScaler().fit(flatten(X_train))

X_train_scaled = scale(X_train, scaler)
X_valid_scaled = scale(X_valid, scaler)


timesteps =  X_train_scaled.shape[1] # equal to the lookback
n_features =  X_train_scaled.shape[2] # 59

epochs = 100
batch_size = 64
lr = 0.0001


lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(32, activation='linear', 
                        input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(4, activation='linear', 
                        return_sequences=False, dropout = 0.5))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(4, activation='linear', 
                    return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='linear', 
                    return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()


adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

# Value Error (flatten 차원 문제 인듯)
lstm_autoencoder_history = lstm_autoencoder.fit(X_train_scaled, y_train, 
                                                epochs=epochs, 
                                                batch_size=batch_size, 
                                                validation_data=(X_valid_scaled, y_valid),
                                                verbose=2).history

plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
