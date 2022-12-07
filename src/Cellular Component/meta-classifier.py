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
df_train = pd.read_csv('data/cc-training.csv')
df_val = pd.read_csv('data/cc-validation.csv')
data_augmentation = pd.read_csv('data/cc-augmented_training.csv')
df_test = pd.read_csv('data/cc-test.csv')
ontologies_names = df_val.columns[2:].values

# Load ontology
ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space='cellular_component')

# Generate slices from sliding window technique
__, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
__, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
__, y_test, positions_test = generate_data(df_test, subseq=500, overlap=250)
__, y_aug, positions_aug = generate_data(data_augmentation, subseq=500, overlap=250)

X_train = np.load('/data/embedding/cc-X_train.npy')
X_val = np.load('/data/embedding/cc-X_val.npy')
X_test = np.load('/data/embedding/cc-X_test.npy')

positions_train = positions_train + positions_aug
y_train = np.concatenate([y_train, y_aug], axis=0)


# Embeddings aggregation
X_train, y_train = protein_embedding(X_train, y_train, positions_train)
X_val, y_val = protein_embedding(X_val, y_val, positions_val)
X_test, y_test = protein_embedding(X_test, y_test, positions_test)


# Meta-classifier fit
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_shape=(X_train.shape[1], ), activation='relu'))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

es = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, lr])

pred = model.predict(X_test)
evaluate(pred, y_test)
np.save('predictions/cc-temprot.npy', pred)