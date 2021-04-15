from math import sqrt
import glob
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import concat
from sklearn.preprocessing import StandardScaler
from numpy import concatenate

path= r'C:\Users\santo\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\fil-main\csv'
filenames = glob.glob(path + "/*.csv")
ttime = []
X1 = []
X2 = []
X3 = []

DX12 = []
DX23 = []

KAVG = []


DV12 = []
DV23 = []

A1 = []
A2 = []
A3 = []

STATE = []
Q = []
V_SHOCKWAVE = []

for filename in filenames:
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
    V_shockwave = []

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

    for k in kavg:
        V_shockwave.append(q / k)
    V_SHOCKWAVE += V_shockwave

# shift = 5
#
# shifted_VSW = [float("nan")] * shift #metatopish thesis data
# shifted_VSW += V_SHOCKWAVE
# shifted_VSW = shifted_VSW[:-(shift)]
#
# df = pd.DataFrame(list(zip(DX12, DX23, DV12, DV23, KAVG, Q, V_SHOCKWAVE, shifted_VSW)))
# df = df.iloc[shift:]
#
# mms = MinMaxScaler()
# df = mms.fit_transform(df)
#
# df = pd.DataFrame(df)
# df_y = pd.DataFrame(df[7])
#
# df_train = df.iloc[:int(len(df) * 0.8)]
# df_test = df.iloc[int(len(df) * 0.8):]
# df_y_train = df_y.iloc[:int(len(df_y)*0.8)]
# df_y_test = df_y.iloc[int(len(df_y)*0.8):]
#
# df_train = np.array(df_train.values)
# df_test = np.array(df_test.values)
# df_y_train = np.array(df_y_train.values)
# df_y_test = np.array(df_y_test.values)
#
# df_train = df_train.reshape((df_train.shape[0], 1, df_train.shape[1]))
# df_test = df_test.reshape((df_test.shape[0], 1, df_test.shape[1]))
#
# model = Sequential()
# model.add(LSTM(50, input_shape=(df_train.shape[1], df_train.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(df_train, df_y_train, epochs=30, batch_size=32, validation_data=(df_test, df_y_test), verbose=2, shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
#
# yhat = model.predict(df_test)
# df_test = df_test.reshape((df_test.shape[0], df_test.shape[2]))
#
# inv_yhat = concatenate((yhat, df_test[:, 1:]), axis=1)
# inv_yhat = mms.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
#
# df_y_test = df_y_test.reshape((len(df_y_test), 1))
# inv_y = concatenate((df_y_test, df_test[:, 1:]), axis=1)
# inv_y = mms.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
#
# pyplot.plot(inv_y, color='red', label='Y')
# pyplot.plot(inv_yhat, color='blue', label='Y hat')
# pyplot.legend()
# pyplot.show()

fnr = []

inaccuracy = []
DR = []
accura = []
accuracy = []
tp=0
tn=0
fp=0
fn=0
index= []

for shift in range(20,200,20):
    STATE_shift = []
    STATE_shift = [float("nan")] * shift
    STATE_shift += STATE
    STATE_shift = STATE_shift[:-(shift)]

    LABELS = pd.DataFrame(list(zip(DX12, DX23, DV12, DV23, KAVG, Q, STATE, STATE_shift)))
    LABELS = LABELS.iloc[shift:]


    print("Shift is now: ", shift)

    mms1 = MinMaxScaler()
    LABELS = mms1.fit_transform(LABELS)

    LABELS = pd.DataFrame(LABELS)
    df_y = pd.DataFrame(LABELS[7])

    df_train = LABELS.iloc[:int(len(LABELS) * 0.8)]
    df_test = LABELS.iloc[int(len(LABELS) * 0.8):]
    df_y_train = df_y.iloc[:int(len(df_y) * 0.8)]
    df_y_test = df_y.iloc[int(len(df_y) * 0.8):]



    df_train = np.array(df_train.values)
    df_test = np.array(df_test.values)
    df_y_train = np.array(df_y_train.values)
    df_y_test = np.array(df_y_test.values)

    df_train = df_train.reshape((df_train.shape[0], 1, df_train.shape[1]))
    df_test = df_test.reshape((df_test.shape[0], 1, df_test.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(df_train.shape[1], df_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(df_train, df_y_train, epochs=30, batch_size=32, validation_data=(df_test, df_y_test), verbose=0,
                        shuffle=False)
    # plot history
    title = "Shift : {}".format(shift)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

    yhat = model.predict(df_test)
    df_test = df_test.reshape((df_test.shape[0], df_test.shape[2]))

    inv_yhat = concatenate((yhat, df_test[:, 1:]), axis=1)
    #inv_yhat = mms1.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    df_y_test = df_y_test.reshape((len(df_y_test), 1))
    inv_y = concatenate((df_y_test, df_test[:, 1:]), axis=1)
    #inv_y = mms1.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # for i in range(len(inv_yhat)):
    #     if inv_yhat[i] > 0.8:
    #         inv_yhat[i]=1
    #     elif inv_yhat[i]<0.2:
    #         inv_yhat[i]=0

    for i in range(len(inv_yhat)):
        accura.append((abs(inv_y[i]-inv_yhat[i])))
    sum_acc = sum(accura)


    inaccuracy.append(sum_acc/len(inv_yhat))

    thres = 0.95
    accuracy.append(1 - (sum_acc / len(inv_yhat)))
    for i in range(len(inv_y)):
        if inv_yhat[i] < thres and df_y_test[i] == 0:
            tn = tn + 1
        elif inv_yhat[i] > 1 - thres and df_y_test[i] == 0:
            fp = fp + 1
        elif inv_yhat[i] < thres and df_y_test[i] == 1:
            fn = fn + 1
        elif inv_yhat[i] > 1 - thres and df_y_test[i] == 1:
            tp = tp + 1
    DR.append(tp/(fn+tp))
    index.append(shift*0.2)

    pyplot.plot(inv_y, color='red', label='Y')
    pyplot.plot(inv_yhat, color='blue', label='Y hat')
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

    print(tp)
    print(tn)
    print(fp)
    print(fn)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    sum_acc = 0

plt.plot(index,accuracy,'o')
plt.xlabel("persistence")
plt.ylabel("accuracy")
plt.show()

plt.plot(index,accuracy)
plt.xlabel("persistence")
plt.ylabel("accuracy")
plt.show()

plt.plot(DR,accuracy,'o')
plt.xlabel("DR")
plt.ylabel("accuracy")
plt.show()

plt.plot(DR,accuracy,)
plt.xlabel("DR")
plt.ylabel("accuracy")
plt.show()

plt.plot(inaccuracy,accuracy,'o')
plt.xlabel("inaccuracy")
plt.ylabel("accuracy")
plt.show()



fnr1 = []
inaccuracy1 = []
DR1 = []
accura11 = []
accuracy11 = []
tp=0
tn=0
fp=0
fn=0
index1= []

for batch_n in range(4,128,8):
    shift=5
    STATE_shift = []
    STATE_shift = [float("nan")] * shift
    STATE_shift += STATE
    STATE_shift = STATE_shift[:-(shift)]

    LABELS = pd.DataFrame(list(zip(DX12, DX23, DV12, DV23, KAVG, Q, STATE, STATE_shift)))
    LABELS = LABELS.iloc[shift:]


    print("persistence is now: ", batch_n)

    mms1 = MinMaxScaler()
    LABELS = mms1.fit_transform(LABELS)

    LABELS = pd.DataFrame(LABELS)
    df_y = pd.DataFrame(LABELS[7])

    df_train = LABELS.iloc[:int(len(LABELS) * 0.8)]
    df_test = LABELS.iloc[int(len(LABELS) * 0.8):]
    df_y_train = df_y.iloc[:int(len(df_y) * 0.8)]
    df_y_test = df_y.iloc[int(len(df_y) * 0.8):]



    df_train = np.array(df_train.values)
    df_test = np.array(df_test.values)
    df_y_train = np.array(df_y_train.values)
    df_y_test = np.array(df_y_test.values)

    df_train = df_train.reshape((df_train.shape[0], 1, df_train.shape[1]))
    df_test = df_test.reshape((df_test.shape[0], 1, df_test.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(df_train.shape[1], df_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(df_train, df_y_train, epochs=30, batch_size=batch_n, validation_data=(df_test, df_y_test), verbose=0,
                        shuffle=False)
    # plot history
    title = "batch size: {}".format(batch_n)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

    yhat = model.predict(df_test)
    df_test = df_test.reshape((df_test.shape[0], df_test.shape[2]))

    inv_yhat = concatenate((yhat, df_test[:, 1:]), axis=1)
    #inv_yhat = mms1.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    df_y_test = df_y_test.reshape((len(df_y_test), 1))
    inv_y = concatenate((df_y_test, df_test[:, 1:]), axis=1)
    #inv_y = mms1.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # for i in range(len(inv_yhat)):
    #     if inv_yhat[i] > 0.8:
    #         inv_yhat[i]=1
    #     elif inv_yhat[i]<0.2:
    #         inv_yhat[i]=0

    for i in range(len(inv_yhat)):
        accura11.append((abs(inv_y[i]-inv_yhat[i])))
    sum_acc = sum(accura)


    inaccuracy1.append(sum_acc/len(inv_yhat))

    thres = 0.9
    accuracy11.append(1 - (sum_acc / len(inv_yhat)))
    for i in range(len(inv_y)):
        if inv_yhat[i] < thres and df_y_test[i] == 0:
            tn = tn + 1
        elif inv_yhat[i] > 1 - thres and df_y_test[i] == 0:
            fp = fp + 1
        elif inv_yhat[i] < thres and df_y_test[i] == 1:
            fn = fn + 1
        elif inv_yhat[i] > 1 - thres and df_y_test[i] == 1:
            tp = tp + 1
    DR1.append(tp/(fn+tp))
    index1.append(shift*0.2)
    fnr1.append(fn/(fn+tn))
    pyplot.plot(inv_y, color='red', label='Y')
    pyplot.plot(inv_yhat, color='blue', label='Y hat')
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

    print(tp)
    print(tn)
    print(fp)
    print(fn)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    sum_acc = 0

plt.plot(index1,accuracy11,'o')
plt.xlabel("persistence")
plt.ylabel("accuracy")
plt.show()

plt.plot(index1,accuracy11)
plt.xlabel("persistence")
plt.ylabel("accuracy")
plt.show()

plt.plot(DR1,fnr1,'o')
plt.xlabel("DR")
plt.ylabel("fnr")
plt.show()

plt.plot(DR1,accuracy11)
plt.xlabel("DR")
plt.ylabel("accuracy")
plt.show()
# Compare element-wise
# for x, y in zip(listA, listB):
#     if x < y: ... # Your code