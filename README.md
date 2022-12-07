# TEMPROT and DS-TEMPROT: protein function annotation using Transformers embeddings and homology search

## Introduction
This work is under review process.

## Dataset
The dataset of this work can be found [here](https://zenodo.org/record/7409660).

## Reproducibility
Run the following sequence of files for each ontology:
```
1. data-augmentation.py
2. fine-tuning.py
3. extract-embeddings.py
4. meta-classifier.py
5. ds-temprot.py
```
To generate the database used in this work, download DeepGOPlus CAFA3 dataset [here](https://deepgo.cbrc.kaust.edu.sa/data/) and run ```dataset.py``` file for each ontology.

