

import math
from typing import List, Any, Union

from scipy.optimize import curve_fit
from scipy import interpolate
import glob
import pandas as pd
import matplotlib
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras import backend as K
from keras.layers import Dense

from keras.models import Sequential
from matplotlib import pyplot

path= r'C:\Users\santo\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\fil-main\csv'
filenames = glob.glob(path + "/*.csv")
ttime = []
X1 = []
X2 = []
X3 = []

DX12 = []
DX23 = []

KAVG = []
q = (0.1923+0.658+0.595+0.134+0.51+0.2099+0.345)/7

DV12 = []
DV23 = []

A1 = []
A2 = []
A3 = []

STATE = []
Q = []


for filename in filenames:
    print(filename)
    colnames = ['Time', 'X1', 'X2', 'X3', 'state']
    df = pd.read_csv(filename, sep=",", header=None, skiprows=2, names=colnames, index_col=False)
    # dt_check = 1
    time = []
    x1 = []
    x2 = []
    x3 = []
    state = []
    dx12 = []
    dx23 = []
    dv12 = []
    dv23 = []
    da12 = []
    da23 = []
    v1 = []
    v2 = []
    v3 = []
    a1 = []
    a2 = []
    a3 = []
    kavg = []
    q_list = []

    df = df.fillna(0)
    df = df.replace(['S', 'M', 'F', 's', 'm', 'f'], 1)

    for ind in df.index:
        time.append(float(str(df['Time'][ind]).replace(",", ".")))
        x1.append(float(str(df['X1'][ind]).replace(",", ".")))
        x2.append(float(str(df['X2'][ind]).replace(",", ".")))
        x3.append(float(str(df['X3'][ind]).replace(",", ".")))
        state.append(float(df['state'][ind]))

    STATE += state

    for a, b in zip(x1, x2):
        dx12.append(abs(a - b))
    DX12 += dx12

    for a, b in zip(x3, x2):
        dx23.append(abs(a - b))
    DX23 += dx23

    # taxythtes oxhmatwn
    for i in range(len(x1) - 1):
        v1.append((x1[i] - x1[i+1]) / 0.2)
    v1.append(v1[-1])

    for i in range(len(x2) - 1):
        v2.append((x2[i] - x2[i+1]) / 0.2)
    v2.append(v2[-1])

    for i in range(len(x3) - 1):
        v3.append((x3[i] - x3[i+1]) / 0.2)
    v3.append(v3[-1])

    for a, b in zip(v1, v2):
        dv12.append(abs(a - b))
    DV12 += dv12

    for a, b in zip(v2, v3):
        dv23.append(abs(a - b))
    DV23 += dv23

    # epitaxynseis oxhmatwn
    for i in range(len(v1) - 1):
        a1.append((v1[i] - v1[i+1]) / 0.2)
    a1.append(a1[-1])
    A1 += a1

    for i in range(len(v2) - 1):
        a2.append((v2[i] - v2[i+1]) / 0.2)
    a2.append(a2[-1])
    A2 += a2

    for i in range(len(v3) - 1):
        a3.append((v3[i] - v3[i+1]) / 0.2)
    a3.append(a3[-1])
    A3 += a3

    for a, b in zip(dx12, dx23):
        kavg.append((a + b) / 2)
    KAVG += kavg

    # ypologismos q
    x1_p0 = 0
    x3_p0 = 0

    for i in range(len(x1)):
        if x1[i] >= 0:
            x1_p0 = i
            break
        else:
            i += 1

    for i in range(len(x3)):
        if x3[i] >= 0:
            x3_p0 = i
            break
        else:
            i += 1

    x1_p0 = int(len(x1) / 2)
    mid = x1[x1_p0]
    for i, x in enumerate(x3):
        if x > mid:
            x3_p0 = i
            break

    # roh
    q = 3 / abs(x1_p0 - x3_p0) * 0.2
    q_list = [q] * len(x1)
    Q += q_list

X = zip(DX12, DX23, DV12, DV23, KAVG, Q)

print(len(DX12))

mms = MinMaxScaler()
X = mms.fit_transform(list(X))

# size = int(len(X) * 0.80)
#
# trainX, testX = X[0:size, :], X[size:len(X), :]
# trainY, testY = STATE[0:size], STATE[size:len(X)]
#
# trainX = pd.DataFrame(trainX).to_numpy()
# trainY = pd.DataFrame(trainY).to_numpy()
#
# testX = pd.DataFrame(testX).to_numpy()
# testY = pd.DataFrame(testY).to_numpy()

# model = Sequential()
# model.add(Dense(15, activation="relu", input_dim=6))
# model.add(Dense(25, activation="relu", input_dim=15))
# model.add(Dense(1, activation="sigmoid", input_dim=25))
#
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(trainX, trainY, epochs=100, batch_size=4, verbose=True)
#
# scores = model.evaluate(testX, testY, verbose=0)
# preds = model.predict_classes(testX)

accuracy_list = []
# K Fold
X = pd.DataFrame(X).to_numpy()
STATE = pd.DataFrame(STATE).to_numpy()

kfold = KFold(n_splits=3, shuffle=True)
for i, (train, test) in enumerate(kfold.split(X)):

    model = Sequential()
    model.add(Dense(150, activation="relu", input_dim=6))
    model.add(Dense(75, activation="relu", input_dim=150))
    model.add(Dense(1, activation="sigmoid", input_dim=75))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X[train], STATE[train], epochs=50, batch_size=4, verbose=True)

    scores = model.evaluate(X[test], STATE[test], verbose=1)
    accuracy_list.append(scores)
    #print(metrics.accuracy_score((X[test], STATE[test]))

print(accuracy_list)

#plt.plot(history.history['loss'], color='blue', label='Loss')
#plt.title("NN - Metrics")
#plt.legend()
#plt.show()

y_pred = model.predict(X)



threshold = 0.3708

ny_pred =[]
for i in range(0,len(y_pred)):
    ny_pred.append(y_pred[i])
predictions_binary = []

any_pred = []
for i in range(0,len(y_pred)):
    any_pred.append(y_pred[i])

for p in y_pred:
    if p > threshold:
        predictions_binary.append(1)
    else:
        predictions_binary.append(0)

cm = confusion_matrix(STATE,predictions_binary  )

fig, model = plt.subplots(figsize=(8, 8))
model.imshow(cm)
model.grid(False)
model.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
model.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
model.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        model.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


plt.plot(STATE, color='green', label='Given State')
plt.plot(predictions_binary, color='blue', label='Predicted state - Binary Classification')
plt.title('Model Predictions')
plt.legend()
plt.show()


ns_probs = [0 for _ in range(len(STATE))]
lr_probs = predictions_binary

ns_auc = roc_auc_score(STATE, ns_probs)
lr_auc = roc_auc_score(STATE, lr_probs)
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Logistic: ROC AUC=%.3f' % lr_auc)
ns_fpr, ns_tpr, _ = roc_curve(STATE, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(STATE, y_pred)
# find optimal thresholds
lr = lr_tpr - lr_fpr
optimal_idx = np.argmax(lr)
optimal_threshold = _[optimal_idx]
print('Optimal threshold: ', optimal_threshold)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

dr_ind=0
DR1=[]
FAR1=[]
far_ind=0

fp=0
tp=0
fn=0
tn=0


for i in range(0,50,1):
    threshold = i/100
    for j in range(len(ny_pred)):

        if float(ny_pred[j]) > threshold and float(STATE[j]) == 1:
                tp = tp + 1
        elif float(ny_pred[j]) > threshold and float(STATE[j]) == 0:
                fp = fp + 1
        elif float(ny_pred[j]) < threshold and float(STATE[j]) == 0:
                tn = tn + 1
        elif float(ny_pred[j]) < threshold and float(STATE[j]) == 1:
                fn = fn + 1





    DR1.append(tp/ (tp+fn))
    FAR1.append(fp/(fp+fn+tn+tp))
    tp =0
    fp=0
    tn=0
    fn=0
print(DR1)
print(FAR1)






for i in range(50,100,1):
    threshold = i/100
    for j in range(len(any_pred)):
        if float(any_pred[j]) > threshold and float(STATE[j]) == 1:
            tp = tp + 1
        elif float(any_pred[j]) < threshold and float(STATE[j]) == 1:
            fn = fn + 1
        elif float(any_pred[j]) > threshold and float(STATE[j]) == 0:
            fp = fp + 1

        elif float(any_pred[j]) < threshold and float(STATE[j]) == 0:
            tn = tn + 1

    print(tp)
    print(fn)
    DR1.append(tp/ (tp+fn))
    FAR1.append(fp/len(any_pred))
    dr_ind=0
    far_ind=0
    fp=0
    tp=0
    fn=0
    tn=0
print(DR1)
print(FAR1)

plt.plot(FAR1,DR1,'o')
plt.xlabel("FAR")
plt.ylabel("DR")
plt.show()

plt.plot(FAR1,DR1)
plt.xlabel("FAR")
plt.ylabel("DR")
plt.show()