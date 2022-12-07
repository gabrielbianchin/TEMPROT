import pandas as pd
import numpy as np
from tqdm import tqdm

exec(open('data/utils.py').read())

ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space='molecular_function')
train = pd.read_pickle('train_data_train.pkl')
valid = pd.read_pickle('train_data_valid.pkl')
test = pd.read_pickle('test_data.pkl')
terms = pd.read_pickle('terms.pkl')

# Function to preprocess the dataset
def preprocess(df):
    f_df = {}
    f_df['proteins'] = []
    f_df['sequences'] = []
  
    for i in list_terms:
        f_df[i] = []
    
    for i in tqdm(range(len(df))):
        
        is_in_dataset = False
        actual_terms = [0 for _ in range(len(list_terms))]
        
        protein, sequence, annotation = df.iloc[i, :].values

        for term in list(annotation):
            if term in list_terms:
                is_in_dataset = True
                actual_terms[list_terms.index(term)] = 1
                for ant in ontology[term]['ancestors']:
                    actual_terms[list_terms.index(ant)] = 1

        if is_in_dataset:
            f_df['proteins'].append(protein)
            f_df['sequences'].append(sequence)
            
            for j in range(len(list_terms)):
                f_df[list_terms[j]].append(actual_terms[j])
    
    return pd.DataFrame(f_df)


list_terms = []
for i in terms.terms.tolist():
    if i in ontology:
        list_terms.append(i)

# Generate new sets
n_train = preprocess(train)
n_val = preprocess(valid.iloc[:, :3])
n_test= preprocess(test)

# Remove duplicated from
# 1 - training and test, removing from training
# 2 - validation and test, removing from validation
# 3 - validation and training, removing from validaiton
n_train = n_train[~n_train.sequences.isin(n_test.sequences.values)]
n_val = n_val[~n_val.sequences.isin(n_test.sequences.values)]
n_val = n_val[~n_val.sequences.isin(n_train.sequences.values)]

# Save
n_train.to_csv('mf-training.csv', index=False)
n_val.to_csv('mf-validation.csv', index=False)
n_test.to_csv('mf-testing.csv', index=False)