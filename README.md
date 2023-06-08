# TEMPROT: protein function annotation using transformers embeddings and homology search

## Introduction
Although the development of sequencing technologies has provided a large number of protein sequences, the analysis of functions that each one plays is still difficult due to the efforts of laboratorial methods, making necessary the usage of computational methods to decrease this gap. As the main source of information available about proteins is their sequences, approaches that can use this information, such as classification based on the patterns of the amino acids and the inference based on sequence similarity using alignment tools, are able to predict a large collection of proteins. The methods available in the literature that use this type of feature can achieve good results, however, they present restrictions of protein length as input to their models. In this work, we present a new method, called TEMPROT, based on the fine-tuning and extraction of embeddings from an available architecture pre-trained on protein sequences. We also describe TEMPROT+, an ensemble between TEMPROT and BLASTp, a local alignment tool that analyzes sequence similarity, which improves the results of our former approach.

The paper can be found [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05375-0).

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

## Citation
```
@article{ijms222111449,
AUTHOR = {Oliveira, Gabriel B. and Pedrini, Helio and Dias, Zanoni},
TITLE = {TEMPROT: protein function annotation using transformers embeddings and homology search},
JOURNAL = {BMC Bioinformatics},
VOLUME = {24},
YEAR = {2023},
NUMBER = {242},
URL = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05375-0},
ISSN = {1471-2105},
DOI = {10.1186/s12859-023-05375-0}
}
```
