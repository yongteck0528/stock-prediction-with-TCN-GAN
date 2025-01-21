import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def sliding_window(x, y, window):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    
    x_ = np.array(x_, dtype=np.float32)
    y_ = np.array(y_, dtype=np.float32)
    y_gan = np.array(y_gan, dtype=np.float32)
    
    return x_, y_, y_gan


df = pd.read_csv('dataset/ITMG.JK.csv')

features = ['Open', 'High', 'Low', 'Close']
target = 'Close'

df = df.dropna()

X = df[features].values
y = df[target].values.reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1,1))

window_size = 10
train_x_slide, train_y_slide, train_y_gan = sliding_window(X_train_scaled, y_train_scaled, window_size)
test_x_slide, test_y_slide, test_y_gan = sliding_window(X_test_scaled, y_test_scaled, window_size)