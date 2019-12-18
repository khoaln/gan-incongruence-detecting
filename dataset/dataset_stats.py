import os

FILENAMES = ['train.csv', 'dev.csv', 'test.csv']

for file in FILENAMES:
  with open(os.path.join('output_full', file), 'r') as f:
    print(file)
    congruent, incongruent = 0, 0
    for line in f:
      line = line.split(',')
      label = int(line[-1])
      if label == 0:
        incongruent += 1
      if label == 1:
        congruent += 1
    print('Congruent:', congruent)
    print('Incongruent:', incongruent)