import tensorflow as tf
from tensorflow.keras import layers
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# load BERT model
bert = pickle.load(open("dataset/bert.model", 'rb'))
doc2vec = Doc2Vec.load("dataset/d2v.model")

def loadData2(file):
  data = []
  classes = []
  with open(file, 'r') as f:
    for line in f:
      line = line.split(',')
      id = int(line[0])
      orig_headline_embed = doc2vec.docvecs["{}_headline".format(id)]
      gen_headline_embed = doc2vec.docvecs["{}_generated_headline".format(id)]
      similarity = doc2vec.docvecs.n_similarity(["{}_headline".format(id)], 
                                              ["{}_generated_headline".format(id)])
      data.append(np.concatenate((orig_headline_embed, [similarity], gen_headline_embed), axis=None))
      classes.append(int(line[-1]))
  return data, classes

def loadData(file):
  data = []
  classes = []
  with open(file, 'r') as f:
    for line in f:
      line = line.split(',')
      id = int(line[0])
      orig_headline_embed = bert[id*2].reshape(1, 1024)
      gen_headline_embed = bert[id*2 + 1].reshape(1, 1024)
      similarity = cosine_similarity(orig_headline_embed, gen_headline_embed)[0][0]
      data.append(np.concatenate((orig_headline_embed, [similarity], gen_headline_embed), axis=None))
      classes.append(int(line[-1]))
  return data, classes

# load data
train_data, train_classes = loadData('dataset/output_bert_similarity/train.csv')
test_data, test_classes = loadData('dataset/output_bert_similarity/test.csv')

train_data = np.array(train_data)
train_classes = np.array(train_classes)
test_data = np.array(test_data)
test_classes = np.array(test_classes)

model = tf.keras.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(2049,)))
model.add(layers.Dense(100))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_classes, epochs=90, batch_size=500)

loss, accuracy = model.evaluate(test_data, test_classes)
# loss, accuracy = model.evaluate(np.array(train_data), np.array(train_classes))

predict = model.predict_proba(test_data)
predict_result = []
for p in predict:
  predict_result.append(0 if p[0] > p[1] else 1)

fpr, tpr, thresholds = roc_curve(test_classes, predict_result)

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr)
# print(predict[0])
auc_score = roc_auc_score(test_classes, predict_result)

print('Testing loss: %0.2f' %loss)
print('Testing accuracy: %0.2f%%' % (accuracy*100))
print('AUC: %0.2f%%' % (auc_score*100))