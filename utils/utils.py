import ktrain
from ktrain import text

#### LOAD ONTOLOGY ####
def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []
  
  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)
  
  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True
        
      elif line == '[Typedef]':
        flag = False
      
      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False
    
    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]
    
    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)
    
  return ontology

def get_and_print_children(ontology, term):
  children = {}
  if term in ontology:
    for i in ontology[term]['children']:
      children[i] = ontology[i]
      print(i, ontology[i]['name'])
  return children



#### GENERATE DATA BASED ON SLIDING WINDOW TECHNIQUE ####
def generate_data(df, subseq=100, overlap=0):
  X = []
  y = []
  positions = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):

    len_seq = int(np.ceil(len(sequences[i]) / subseq))

    for idx in range(len_seq):
      if idx != len_seq - 1:
        X.append(' '.join(list(sequences[i][idx * subseq : (idx + 1) * subseq])))
      else:
        X.append(' '.join(list(sequences[i][idx * subseq : ])))
      positions.append(i)
      y.append(df.iloc[i, 2:])

    if overlap > 0:
      init = overlap
      while True:
        if init + subseq >= len(sequences[i]):
          break
        X.append(' '.join(list(sequences[i][init : init + subseq])))
        positions.append(i)
        y.append(df.iloc[i, 2:])
        init += subseq
  
  return X, np.array(y, dtype=int), positions



#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

def get_model(model_name, model_path):
  t = text.Transformer(model_name, maxlen=MAX_LEN, classes=ontologies_names)
  trn = t.preprocess_train(X_train, y_train)
  model = t.get_classifier()
  model.load_weights(model_path)
  learner = ktrain.get_learner(model, batch_size=4, train_data=trn)
  predictor = ktrain.get_predictor(learner.model, preproc=t)
  learner.model.save_pretrained('/weights/full-model')
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  return TFAutoModel.from_pretrained('/weights/full-model'), tokenizer

def get_embeddings(X, model, tokenizer, file_path, last_saved_i, last_saved_emb):
  emb = last_saved_emb.tolist()
  for i in tqdm(range(last_saved_i+1, len(X))):
    input_ids = tf.constant(tokenizer.encode(X[i]))[None, :]
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb.append(last_hidden_states.numpy()[0, 0, :])
    if i % 1000 == 0:
      save_numpy(file_path, np.array(emb, dtype='float64'))

  embedding = np.array(emb, dtype='float64')
  save_numpy(file_path, embedding)

  return embedding




#### COMBINE EMBEDDINGS ####
def protein_embedding(X, y, pos):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
  n_y = [y[0]]
    
  for i in range(len(pos)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(np.mean(np.array(cur_emb), axis=0))
      last_pos = cur_pos
      cur_emb = [X[i]]
      n_y.append(y[i])

  n_X.append(np.mean(np.array(cur_emb), axis=0))
    
  return np.array(n_X), np.array(n_y)