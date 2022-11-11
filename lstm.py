import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, InputLayer


person = pd.read_csv("ramp_walk_2.csv")
person.drop_duplicates(subset="frame", keep=False, inplace=True)
person_work = person.copy()
person_work.drop(['time', 'person_id', 'Unnamed: 0'], axis=1, inplace=True)
person_work.set_index(['frame'], inplace=True)
person_focus = person_work[['left_ankle_x', 'left_ankle_y', 'left_knee_x',
                            'left_knee_y', 'right_ankle_x', 'right_ankle_y','right_knee_x', 'right_knee_y']]


scalar = MinMaxScaler()
person_focus_scaled = scalar.fit_transform(person_focus)


def df_to_x_y(df, window_size=10):
    df_as_np = df
    x = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        x.append(row)
        label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1], df_as_np[i+window_size][2], df_as_np[i+window_size]
                 [3], df_as_np[i+window_size][4], df_as_np[i+window_size][5], df_as_np[i+window_size][6], df_as_np[i+window_size][7]]
        y.append(label)
    return np.array(x), np.array(y)


x, y = df_to_x_y(person_focus_scaled)

x_train = x[:int(len(x)*0.9)]
y_train = y[:int(len(y)*0.9)]
x_val = x[int(len(x)*0.9):-10]
y_val = y[int(len(y)*0.9):-10]
x_test = x[-10:]
y_test = y[-10:]


model = Sequential()

model.add(LSTM(64, input_shape=(10, 8), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, ))
model.add(Dense(34))
model.add(Dense(8))

cp = ModelCheckpoint('model/', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=Adam(
    learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=64, epochs=200, callbacks=[cp])

model = load_model('model')


def plot_predictions(model, X, y):
    predictions = model.predict(X)
    pred_unscaled = scalar.inverse_transform(predictions)
    l_a_x_preds, l_a_y_preds = pred_unscaled[:, 0], pred_unscaled[:, 1]
    l_k_x_preds, l_k_y_preds = pred_unscaled[:, 2], pred_unscaled[:, 3]
    r_a_x_preds, r_a_y_preds = pred_unscaled[:, 4], pred_unscaled[:, 5]
    r_k_x_preds, r_k_y_preds = pred_unscaled[:, 6], pred_unscaled[:, 7]
    df = pd.DataFrame(data={'Left_Ankle_X Predictions': l_a_x_preds,
                            'Left_Ankle_Y Predictions': l_a_y_preds,
                            'Left_Knee_X Predictions': l_k_x_preds,
                            'Left_Knee_Y Predictions': l_k_y_preds,
                            'Right_Ankle_X Predictions': r_a_x_preds,
                            'Right_Ankle_Y Predictions': r_a_y_preds,
                            'Right_Knee_X Predictions': r_k_x_preds,
                            'Right_Knee_Y Predictions': r_k_y_preds,
                            })

    return df[:]

plot_predictions(model, x_test, y_test)