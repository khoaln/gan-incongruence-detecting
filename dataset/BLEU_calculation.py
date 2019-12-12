import nltk
import pandas as pd
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def calculate_BLEU(body, headline):
  body_sentences = sent_tokenize(body)
  hypothesis = word_tokenize(headline)
  references = []
  for s in body_sentences:
    references.append(word_tokenize(s))
  
  return nltk.translate.bleu_score.corpus_bleu([references], [hypothesis])

BASE_PATH = 'log_gorina4/decode'

column_labels = ['id', 'bleu_original', 'bleu_generated']
df = pd.DataFrame(columns=column_labels)
bleu_original = 0
bleu_generated = 0

i = 0
for article_filename in sorted(os.listdir(os.path.join(BASE_PATH, 'article'))):
  i += 1
  if i%1000 == 0:
    print(i)
  id = article_filename.split('_')[0]
  decoded_filename = "{}_decoded.txt".format(id)
  reference_filename = "{}_reference.txt".format(id)
  article_dict = {
    'id': id,
    'bleu_original': 0,
    'bleu_generated': 0
  }

  with open(os.path.join(BASE_PATH, 'article', article_filename)) as f:
    body = f.read()
  with open(os.path.join(BASE_PATH, 'decoded', decoded_filename)) as f:
    generated_headline = re.sub(' +', ' ', f.read().replace('[UNK]', ''))
  with open(os.path.join(BASE_PATH, 'reference', reference_filename)) as f:
    original_headline = f.read()

  bo = calculate_BLEU(body, original_headline)
  bg = calculate_BLEU(body, generated_headline)
  bleu_original += bo
  bleu_generated += bg
  article_dict['bleu_original'] = bo
  article_dict['bleu_generated'] = bg

  df = df.append(article_dict, ignore_index=True)

df.to_csv('bleu_score.csv')
print("Original BLEU", bleu_original/(i-1))
print("Generated BLEU", bleu_generated/(i-1))
  

