from sklearn import svm
from sklearn.metrics import accuracy_score

X_train = []
y_train = []
X_test = []
y_test = []

with open('dataset/output/train.csv', 'r') as f:
  for line in f:
    line = line.replace('\n', '').split(",")
    X_train.append([line[-2]])
    y_train.append(line[-1])

with open('dataset/output/test.csv', 'r') as f:
  for line in f:
    line = line.replace('\n', '').split(",")
    X_test.append([line[-2]])
    y_test.append(line[-1])

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction))