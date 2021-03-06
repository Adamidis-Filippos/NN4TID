import matplotlib.pyplot as plt
from numpy import mean
from numpy import std

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import math
import plot_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import glob
import pandas as pd
from keras.utils.vis_utils import plot_model
import pydot
from matplotlib import pyplot

#Loading data and variables
path= r'C:\Users\santo\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\fil-main\nm'
filenames = glob.glob(path + "/*.csv")
time = []
Y1 = []
Y2 = []
X1 = []
X2 = []
V = []
ACC = []
state = []
DIST = []

for filename in filenames:
    colnames = ['Time', 'Y1', 'Y2', 'X1', 'X2', 'state']
    df = pd.read_csv(filename, sep=",", header=None, skiprows=2, names=colnames, index_col=False)
    dt_check = 1
    time = []
    y1 = []
    y2 = []
    x1 = []
    x2 = []
    v = []
    acc = []
    dist = []
    for ind in df.index:
        time.append(float(str(df['Time'][ind]).replace(",", ".")))
        yy1 = float(str(df['Y1'][ind]).replace(",", "."))
        y1.append(0 if math.isnan(yy1) else yy1)
        yy2 = float(str(df['Y2'][ind]).replace(",", "."))
        y2.append(0 if math.isnan(yy2) else yy2)
        xx1 = float(str(df['X1'][ind]).replace(",", "."))
        x1.append(0 if math.isnan(xx1) else xx1)
        xx2 = float(str(df['X2'][ind]).replace(",", "."))
        x2.append(0 if math.isnan(xx2) else xx2)
        state.append( 0 if math.isnan(df['state'][ind]) else df['state'][ind])

    dt = abs(time[-2] - time[-1])


    for xx1, xx2, yy1, yy2 in zip(x1, x2, y1, y2):
       # print(xx1, xx2, yy1, yy2)
        dist.append( ( (xx2-xx1)**2 + (yy2-yy1)**2 ) ** 0.5 )

    for a, b in zip(dist, dist[1:]):
        v.append(abs(b - a) / dt)
    v.append(v[-1])

    for a, b in zip(v, v[1:]):
        acc.append(abs(b - a) / dt)
    acc.append(v[-1])

    Y1.extend(y1)
    Y2.extend(y2)
    X1.extend(x1)
    X2.extend(x2)
    V.extend(v)
    ACC.extend(acc)
    DIST.extend(dist)





Y1=[]
Y2=[]
X1=[]
X2=[]
STATE1=[]



X = list(zip(DIST, V, ACC))

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# define the model evaluation procedure
model.fit(X,state)
cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, state, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


y_prob = model.predict_proba(X)
y_pred = model.predict(X)

threshold = 0.5
for i in range(len(y_pred)):
    if y_prob[i][1] > threshold:
        y_pred[i] = 1

cm = confusion_matrix(state, y_pred)

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


pyplot.plot(V, state, linestyle='--', label='DIST')
pyplot.plot(DIST, state, marker='.', label='state')
pyplot.xlabel('DIST')
pyplot.ylabel('state')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

plt.plot(y_pred,V,'o')
plt.xlabel("STATE pred")
plt.ylabel("DV")
plt.show()

plt.plot(state,V,'o')
plt.xlabel("STATE")
plt.ylabel("DV")
plt.show()

plt.plot(y_pred,DIST,'o')
plt.xlabel("STATE pred")
plt.ylabel("DIST")
plt.show()

plt.plot(state,DIST,'o')
plt.xlabel("STATE")
plt.ylabel("DIST")
plt.show()

plt.figure(1)
plt.title("Rel. Speed")
plt.plot(V, color= 'red', label = 'Rel. Speed')
plt.xlabel("data")
plt.ylabel("DV")

plt.figure(2)
plt.title("Rel. Dist")
plt.plot(DIST, color= 'yellow', label = 'Rel. Distance')
plt.xlabel("data")
plt.ylabel("DIST")

plt.figure(3)
plt.title("State")
plt.plot(state, color = 'blue', label = 'State')
plt.xlabel("data")
plt.ylabel("Pred state")

plt.show()

ns_probs = [0 for _ in range(len(state))]
lr_probs = y_prob[:, 1]

ns_auc = roc_auc_score(state, ns_probs)
lr_auc = roc_auc_score(state, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Logistic: ROC AUC=%.3f' % lr_auc)
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(state, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(state, lr_probs)
print('_:', _)
# find optimal threshold
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

tp = 0
fp = 0
tn = 0
fn = 0
DR1 = []
FAR1 = []

for i in range(0,50,1):
    threshold = i/100
    for j in range(len(y_pred)):

        if float(y_prob[j][1]) > threshold and float(state[j]) == 1:
                tp = tp + 1
        elif float(y_prob[j][1]) > threshold and float(state[j]) == 0:
                fp = fp + 1
        elif float(y_prob[j][1]) < threshold and float(state[j]) == 0:
                tn = tn + 1
        elif float(y_prob[j][1]) < threshold and float(state[j]) == 1:
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
    for j in range(len(y_prob)):
        if float(y_prob[j][1]) > threshold and float(state[j]) == 1:
            tp = tp + 1
        elif float(y_prob[j][1]) < threshold and float(state[j]) == 1:
            fn = fn + 1
        elif float(y_prob[j][1]) > threshold and float(state[j]) == 0:
            fp = fp + 1

        elif float(y_prob[j][1]) < threshold and float(state[j]) == 0:
            tn = tn + 1

    print(tp)
    print(fn)
    DR1.append(tp/ (tp+fn))
    FAR1.append(fp/len(y_pred))
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

index=1
DR = []
dr_ind=0
for i in range(0,len(y_pred)):
    if y_pred[i] > 0.8 and state[i]== 1 :
        DR.append(1)
        dr_ind = dr_ind +1
        index =index +1
print(len(DR))

DR= dr_ind/len(y_pred)
print(DR)
FAR=[]
far_ind=0
for i in range(len(y_pred)):
    if y_pred[i] == 0 and state[i]!= 0 :
        far_ind= far_ind+1


FAR = far_ind/ len(y_pred)
print(FAR)

ny_pred =[]
for i in range(0,len(y_pred)):
    ny_pred.append(y_pred[i])

dr_ind=0
DR1=[]
FAR1=[]
far_ind=0


for i in np.arange(0.0 ,0.5 ,0.1):
    threshold = i
    for j in range(len(y_pred)):
        if y_prob[j][1] > threshold:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
        if y_pred[j] ==1 and state[j] ==1 :
            dr_ind = dr_ind +1
        elif y_pred[j] ==0 and state[j] ==1 :
            far_ind = far_ind + 1
    DR1.append(dr_ind/ len(y_pred)) #TPR
    FAR1.append(far_ind/len(y_pred))
    dr_ind =0
    far_ind=0
print(DR1)
print(FAR1)

for i in np.arange(0.5 ,1.0 ,0.1):
    threshold = i
    for j in range(len(ny_pred)):
        if y_prob[j][1] < threshold:
            ny_pred[j] = 0
        else:
            ny_pred[j] = 1
        if ny_pred[j] ==1 and state[j] ==1 :
            dr_ind = dr_ind +1
        elif ny_pred[j] ==0 and state[j] ==1 :
            far_ind = far_ind +1
    DR1.append(dr_ind/len(ny_pred) ) #TPR
    FAR1.append(far_ind/len(ny_pred))
    dr_ind=0
    far_ind=0
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


def get_model():
	model1 = LogisticRegression()
	return model1


def evaluate_model(cv):

    # get the model
    model = get_model()
    # evaluate the model
    scores = cross_val_score(model, X, state, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


# calculate the ideal test condition
ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)
# define folds to test
folds = range(2, 12)
# record mean and min/max of each set of results
means, mins, maxs = list(), list(), list()
# evaluate each k value
for k in folds:
    # define the test condition
    cv = KFold(n_splits=k, shuffle=True, random_state=1)
    # evaluate k value
    k_mean, k_min, k_max = evaluate_model(cv)
    # report performance
    print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    # store mean accuracy
    means.append(k_mean)
    # store min and max relative to the mean
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)
# line plot of k mean values with min/max error bars
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# plot the ideal case in a separate color
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
pyplot.xlabel("tested data")
pyplot.ylabel("accuracy")
# show the plot
pyplot.show()
