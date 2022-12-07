import pandas as pd
import numpy as np

# Generate PAM matrix
list_amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
probabilities = np.array([[9867,2,9, 10,3,8, 17, 21,2,6,4,2,6,2, 22, 35, 32,0,2, 18],
[1,9913,1,0,1, 10,0,0, 10,3,1, 19,4,1,4,6,1,8,0,1],
[4,1, 9822, 36,0,4,6,6, 21,3,1, 13,0,1,2, 20,9,1,4,1],
[6,0, 42, 9859,0,6, 53,6,4,1,0,3,0,0,1,5,3,0,0,1],
[1,1,0,0, 9973,0,0,0,1,1,0,0,0,0,1,5,1,0,3,2],
[3,9,4,5,0, 9876, 27,1, 23,1,3,6,4,0,6,2,2,0,0,1],
[10,0,7, 56,0, 35, 9865,4,2,3,1,4,1,0,3,4,2,0,1,2],
[21,1, 12, 11,1,3,7, 9935,1,0,1,2,1,1,3, 21,3,0,0,5],
[1,8, 18,3,1, 20,1,0, 9912,0,1,1,0,2,3,1,1,1,4,1],
[2,2,3,1,2,1,2,0,0, 9872,9,2, 12,7,0,1,7,0,1, 33],
[3,1,3,0,0,6,1,1,4, 22, 9947,2, 45, 13,3,1,3,4,2, 15],
[2,37, 25,6,0, 12,7,2,2,4,1, 9926, 20,0,3,8, 11,0,1,1],
[1,1,0,0,0,2,0,0,0,5,8,4, 9874,1,0,1,2,0,0,4],
[1,1,1,0,0,0,0,1,2,8,6,0,4, 9946,0,2,1,3, 28,0],
[13,5,2,1,1,8,3,2,5,1,2,2,1,1, 9926, 12,4,0,0,2],
[28,11, 34,7, 11,4,6, 16,2,2,1,7,4,3, 17, 9840, 38,5,2,2],
[22,2, 13,4,1,3,2,2,1, 11,2,8,6,1,5, 32, 9871,0,2,9],
[0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0, 9976,1,0],
[1,0,3,0,3,0,1,0,4,1,1,0,0, 21,0,1,1,2, 9945,1],
[13,2,1,1,3,2,2,3,3, 57, 11,1, 17,1,3,2, 10,0,2, 9901]]) / 10000
probabilities = probabilities/probabilities.sum(axis=0,keepdims=1)

# Load data
df_train = pd.read_csv('data/cc-training.csv')
ontologies_names = df_train.columns[2:].values


# Generate new data
K_VALUE = 2

def generate_augmentation(df):
  sample_ids = range(len(df))
  df_final = {'proteins': [], 'sequences': []}
  for term in df.columns[2:].values.tolist():
    df_final[term] = []
  
  for i in tqdm(sample_ids):
    prot = list(df.iloc[i, 1])

    swaps = int(len(prot) * K_VALUE)
    idx = np.random.choice(np.where(np.in1d(prot, list_amino))[0], size=swaps)

    for j in idx:
      prot[j] = np.random.choice(list_amino, p=probabilities[:, list_amino.index(prot[j])].tolist())

    df_final['proteins'].append(0)
    df_final['sequences'].append(''.join(prot))

    for key, value in zip(df.columns[2:].values.tolist(), df.iloc[i, 2:].values.tolist()):
      df_final[key].append(value)
  
  return pd.DataFrame(df_final)

generated_df = {}
for key in df_train.columns.values.tolist():
  generated_df[key] = []
generated_df = pd.DataFrame(generated_df)

generated_df = generated_df.append(generate_augmentation(df_train))

generated_df.to_csv('data/cc-augmented_training.csv', index=False)