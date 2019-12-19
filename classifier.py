from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

X_train = []
y_train = []
X_test = []
y_test = []

with open('dataset/output_bert_similarity/train.csv', 'r') as f:
  for line in f:
    line = line.replace('\n', '').split(",")
    X_train.append([line[-2]])
    y_train.append(int(line[-1]))

with open('dataset/output_bert_similarity/test.csv', 'r') as f:
  for line in f:
    line = line.replace('\n', '').split(",")
    X_test.append([line[-2]])
    y_test.append(int(line[-1]))

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, prediction)

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr)
# print(predict[0])
auc_score = roc_auc_score(y_test, prediction)
print("Accuracy:", accuracy_score(y_test, prediction))
print('AUC: %0.2f%%' % (auc_score*100))