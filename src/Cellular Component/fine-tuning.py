import pandas as pd
import numpy as np
from tqdm import tqdm
import ktrain
from ktrain import text

exec(open('utils/utils.py').read())
MAX_LEN = 500
model_name = 'Rostlab/prot_bert_bfd'

# Load data
df_train = pd.read_csv('data/cc-training.csv')
df_val = pd.read_csv('data/cc-validation.csv')
data_augmentation = pd.read_csv('data/cc-augmented_training.csv')
ontologies_names = df_val.columns[2:].values

# Load ontology
ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space='cellular_component')

# Generate slices from sliding window technique
X_train, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
X_val, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
X_aug, y_aug, positions_aug = generate_data(data_augmentation, subseq=500, overlap=250)

X_train = X_train + X_aug
y_train = np.concatenate([y_train, y_aug], axis=0)
positions_train = positions_train + positions_aug

# Fine-tuning the model
t = text.Transformer(model_name, maxlen=MAX_LEN, classes=ontologies_names)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=4)
learner.autofit(1e-5, 10, early_stopping=1, checkpoint_folder='weights/cc-fine-tuned')