import argparse
import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TEXT = []

FILENAMES = ['train.csv', 'dev.csv', 'test.csv']
stop_words = set(stopwords.words('english'))

def build_model(data_path, body=0):
  for file in FILENAMES:
    with open(os.path.join(data_path, file), 'r') as f:
      for line in f:
        line = line.split(',')
        TEXT.append(" ".join([w for w in word_tokenize(line[1]) if not w in stop_words]))
        if body == 0:
          TEXT.append(" ".join([w for w in word_tokenize(line[3]) if not w in stop_words]))
        else:
          TEXT.append(" ".join([w for w in word_tokenize(line[2]) if not w in stop_words]))

  model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
  sentence_embeddings = model.encode(TEXT)
  return sentence_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    parser.add_argument('--body', default='0')

    args = parser.parse_args()
    if args.body == '1':
      try:
        model = pickle.load(open("bert_body.model", 'rb'))
      except:
        model = build_model(args.data_path, body=1)
        pickle.dump(model, open("bert_body.model", 'wb'))
    else:
      try:
        model = pickle.load(open("bert.model", 'rb'))
      except:
        model = build_model(args.data_path)
        pickle.dump(model, open("bert.model", 'wb'))

      for file in FILENAMES:
        df = pd.read_csv(os.path.join(args.data_path, file), names=['id', 'headline', 'body', 'generated_headline', 'label'])
        df.insert(loc=4, column='similarity', value=0)
        df['similarity'] = df.apply(lambda row: cosine_similarity(model[int(row['id'])*2].reshape(1, 1024),
                                                            model[int(row['id'])*2 + 1].reshape(1, 1024))[0][0], axis=1)
        df.to_csv(os.path.join(args.output_path, file), header=None, index=False)