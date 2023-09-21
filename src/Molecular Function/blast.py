import pandas as pd
from tqdm import tqdm
import os

df_train = pd.read_csv('data/mf-training.csv')
df_val = pd.read_csv('data/mf-validation.csv')
df_test = pd.read_csv('data/mf-test.csv')

def preprocess(df, mode):
  seq = df.sequences.values
  id = 0
  fasta = ''
  for i in tqdm(seq):
    fasta += '>' + str(mode) + '_' + str(id) + '\n'
    id += 1
    fasta += i
    fasta += '\n'
  return fasta

seq_train = preprocess(df_train, 'train')
seq_test = preprocess(df_test, 'test')

with open('reference.fasta', 'w') as f:
  print(seq_train, file=f)

with open('queries.fasta', 'w') as f:
  print(seq_test, file=f)

os.system("makeblastdb -in 'reference.fasta' -dbtype prot")

seq = {}
s=''
k=''
with open('queries.fasta', "r") as f:
  for lines in f:
    if lines[0]==">":
      if s!='':
        seq[k] = s 
        s=''
      k = lines[1:].strip('\n')
    else:
      s+=lines.strip('\n')
seq[k] = s

output = os.popen("blastp -db reference.fasta -query queries.fasta -outfmt '6 qseqid sseqid bitscore' -evalue 0.001").readlines()

test_bits={}
test_train={}
for lines in output:
  line = lines.strip('\n').split()
  if line[0] in test_bits:
    test_bits[line[0]].append(float(line[2]))
    test_train[line[0]].append(line[1])
  else:
    test_bits[line[0]] = [float(line[2])]
    test_train[line[0]] = [line[1]]

preds_score=[]
nlabels = 677

for s in seq:
  probs = np.zeros(nlabels, dtype=np.float32)
  if s in test_bits:
    weights = np.array(test_bits[s])/np.sum(test_bits[s])

    for j in range(len(test_train[s])):
      id = int(test_train[s][j].split('_')[1])
      temp = y_train[id]
      probs+= weights[j] * temp

  preds_score.append(probs)

np.save('predictions/mf-blast.npy)