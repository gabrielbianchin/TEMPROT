import pandas as pd
import numpy as np
from tqdm import tqdm

alpha = 0.6

# Load data
df_test = pd.read_csv('data/cc-test.csv')
y_test = df_test.iloc[:, 2:].values
ontologies_names = df_test.columns[2:].values

# Load ontology
ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space='cellular_component')

# Load predictions
preds_blast = np.load('predictions/cc-blast.npy')
preds_temprot = np.load('predictions/cc-temprot.npy')

# Ensemble predictions
f_pred = []
for i, j in zip(preds_blast, preds_temprot):
  if np.sum(i) != 0:
    f_pred.append(i * (1-alpha) + j * alpha)
  else:
    f_pred.append(j)
f_pred = np.array(f_pred)

# Evaluate
evaluate(f_pred, y_test)