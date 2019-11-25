import argparse
import os
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

HEADLINES = []

FILENAMES = ['train.csv', 'dev.csv', 'test.csv']
stop_words = set(stopwords.words('english'))

def build_model(data_path):
  for file in FILENAMES:
    with open(os.path.join(data_path, file), 'r') as f:
      for line in f:
        line = line.split(',')
        HEADLINES.append(TaggedDocument(words=[w for w in word_tokenize(line[1]) if not w in stop_words], tags=["{}_headline".format(int(line[0]))]))
        HEADLINES.append(TaggedDocument(words=[w for w in word_tokenize(line[3]) if not w in stop_words], tags=["{}_generated_headline".format(int(line[0]))]))

  model = Doc2Vec(dm = 1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)
  model.build_vocab(HEADLINES)
  return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    try:
      model = Doc2Vec.load("d2v.model")
    except:
      model = build_model(args.data_path)

      # Train the model with 20 epochs 
      for epoch in range(20):
        model.train(HEADLINES, epochs=model.epochs, total_examples=model.corpus_count)
        print("Epoch #{} is complete.".format(epoch+1))

      model.save("d2v.model")

    for file in FILENAMES:
      df = pd.read_csv(os.path.join(args.data_path, file), names=['id', 'headline', 'body', 'generated_headline', 'label'])
      df.insert(loc=4, column='similarity', value=0)
      df['similarity'] = df.apply(lambda row: model.docvecs.n_similarity(["{}_headline".format(row['id'])], 
                                                      ["{}_generated_headline".format(row['id'])]), axis=1)
      df.to_csv(os.path.join(args.output_path, file), header=None, index=False)