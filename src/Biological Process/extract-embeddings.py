import pandas as pd
import numpy as np
from tqdm import tqdm
import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf

exec(open('utils/utils.py').read())
MAX_LEN = 500
model_name = 'Rostlab/prot_bert_bfd'

# Load data
df_train = pd.read_csv('data/bp-training.csv')
df_val = pd.read_csv('data/bp-validation.csv')
data_augmentation = pd.read_csv('data/bp-augmented_training.csv')
df_test = pd.read_csv('data/bp-test.csv')
ontologies_names = df_val.columns[2:].values

# Load ontology
ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space='biological_process')

# Generate slices from sliding window technique
X_train, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
X_val, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
X_test, y_test, positions_test = generate_data(df_test, subseq=500, overlap=250)
X_aug, y_aug, positions_aug = generate_data(data_augmentation, subseq=500, overlap=250)

X_train = X_train + X_aug
y_train = np.concatenate([y_train, y_aug], axis=0)
positions_train = positions_train + positions_aug


# Embeddings extraction
model_path = '/weights/bp-fine-tuned/weights-01.hdf5'
model, tokenizer = get_model(model_name, model_path)

################ Training #####################
file_path = '/data/embedding/bp-X_train.npy'
last_saved_i = -1
last_saved_emb = np.array([])

embedding = get_embeddings(X_train, model, tokenizer, file_path, last_saved_i, last_saved_emb)

################ Validation #####################
file_path = '/data/embedding/bp-X_val.npy'
last_saved_i = -1
last_saved_emb = np.array([])

embedding = get_embeddings(X_val, model, tokenizer, file_path, last_saved_i, last_saved_emb)

################ Test #####################
file_path = '/data/embedding/bp-X_test.npy'
last_saved_i = -1
last_saved_emb = np.array([])

embedding = get_embeddings(X_test, model, tokenizer, file_path, last_saved_i, last_saved_emb)