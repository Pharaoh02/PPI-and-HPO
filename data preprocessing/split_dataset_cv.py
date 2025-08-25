#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create training & test datasets.

After running this script, you will get
    - true labels of full dataset
    - masks of each fold indicating the training & test set
i.e.
    store = {
        "annotation": full dataset
        "mask": [
            {
                "train": training mask of 1st fold
                "test": test mask of 1st fold
            },
            ...
        ]
    }
"""
import json
import pickle
from functools import reduce
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import gene2uniprot


'''with open("C:/Users/pharaoh/Desktop/HPODNets-main/config/preprocessing/split_dataset_cv.json") as fp:
    config = json.load(fp)

# load HPO annotations
with open(config["raw_annotation"]) as fp:
    annotation = json.load(fp)

# load old genes
old_genes = set()
with open(config["old_genes"]) as fp:
    for line in fp:
        old_genes.add(line.strip())
# load old mapping of gene id to uniprot id
old_gene2protein = gene2uniprot(config["old_mapping"], gene_column=0, uniprot_column=1)
# get list of old proteins
old_proteins = set()
for old_gene in old_genes:
    for old_protein in old_gene2protein[old_gene]:
        old_proteins.add(old_protein)

# remove newly added proteins
filter_annotation = {term: [protein for protein in annotation[term]
                            if protein in old_proteins]
                     for term in annotation}

# load HPO
ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                  version=config["ontology"]["version"])

# leave only terms in PA
filter_annotation = {term: filter_annotation[term]
                     for term in filter_annotation
                     if ontology[term].ns == 'pa'}

# list of HPO terms
term_list = list(filter_annotation.keys())
# list of proteins
protein_list = list(reduce(lambda a, b: set(a) | set(b),
                           filter_annotation.values()))

# transform HPO annotations to DataFrame like
#           term1   term2   term3
# protein1      1       0       1
# protein2      0       1       0
# protein3      0       0       1
mlb = MultiLabelBinarizer()
df_filter_annotation = pd.DataFrame(mlb.fit_transform(filter_annotation.values()),
                                    columns=mlb.classes_,
                                    index=filter_annotation.keys()).reindex(
                                    columns=protein_list, index=term_list, fill_value=0).transpose()

# object to be stored
# we first insert labels into object
store = {"annotation": df_filter_annotation}

# make up train & test dataset masks for each fold
train_dataset = [dict() for _ in range(5)]
test_dataset = [dict() for _ in range(5)]
for fold in range(5):
    test_proteins = protein_list[fold::5]
    train_proteins = list(set(protein_list) -
                          set(protein_list[fold::5]))
    for term in filter_annotation:
        test_dataset[fold][term] = test_proteins
        train_dataset[fold][term] = train_proteins

# transform train & test masks to DataFrame
store["mask"] = [dict() for _ in range(5)]
for fold in range(5):
    mlb = MultiLabelBinarizer()
    df_train_dataset = pd.DataFrame(mlb.fit_transform(train_dataset[fold].values()),
                                    columns=mlb.classes_,
                                    index=train_dataset[fold].keys()).reindex(
                                    columns=protein_list, index=term_list, fill_value=0).transpose()

    mlb = MultiLabelBinarizer()
    df_test_dataset = pd.DataFrame(mlb.fit_transform(test_dataset[fold].values()),
                                   columns=mlb.classes_,
                                   index=test_dataset[fold].keys()).reindex(
                                   columns=protein_list, index=term_list, fill_value=0).transpose()

    store["mask"][fold]["train"] = df_train_dataset
    store["mask"][fold]["test"] = df_test_dataset

# write to pickle file
with open(config["dataset"], "wb") as fp:
    pickle.dump(store, fp)'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create training & test datasets.

After running this script, you will get
    - true labels of full dataset
    - masks of each fold indicating the training & test set
i.e.
    store = {
        "annotation": full dataset
        "mask": [
            {
                "train": training mask of 1st fold
                "test": test mask of 1st fold
            },
            ...
        ]
    }
"""
import json
import pickle
from functools import reduce
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import gene2uniprot

with open("C:/Users/pharaoh/Desktop/HPODNets-main/config/preprocessing/split_dataset_cv.json") as fp:
    config = json.load(fp)

# load HPO annotations
with open(config["raw_annotation"]) as fp:
    annotation = json.load(fp)

# load old genes
old_genes = set()
with open(config["old_genes"]) as fp:
    for line in fp:
        old_genes.add(line.strip())
# load old mapping of gene id to uniprot id
old_gene2protein = gene2uniprot(config["old_mapping"], gene_column=0, uniprot_column=1)
# get list of old proteins
old_proteins = set()
for old_gene in old_genes:
    for old_protein in old_gene2protein[old_gene]:
        old_proteins.add(old_protein)

# remove newly added proteins
filter_annotation = {term: [protein for protein in annotation[term]
                            if protein in old_proteins]
                     for term in annotation}

# load HPO
ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                  version=config["ontology"]["version"])

# leave only terms in PA
filter_annotation = {term: filter_annotation[term]
                     for term in filter_annotation
                     if ontology[term].ns == 'pa'}

# list of HPO terms
term_list = list(filter_annotation.keys())
# list of proteins
protein_list = list(reduce(lambda a, b: set(a) | set(b),
                           filter_annotation.values()))

# Create mapping dictionaries for proteins and terms
protein_to_idx = {protein: idx for idx, protein in enumerate(protein_list)}
term_to_idx = {term: idx for idx, term in enumerate(term_list)}

# Build sparse matrix for annotations
rows, cols, data = [], [], []
for term, proteins in filter_annotation.items():
    term_idx = term_to_idx[term]
    for protein in proteins:
        protein_idx = protein_to_idx[protein]
        rows.append(protein_idx)
        cols.append(term_idx)
        data.append(1)

# Create sparse matrix
sparse_annotation = sparse.csr_matrix((data, (rows, cols)),
                                      shape=(len(protein_list), len(term_list)),
                                      dtype=np.int8)

# Convert to DataFrame with sparse format
df_filter_annotation = pd.DataFrame.sparse.from_spmatrix(
    sparse_annotation,
    index=protein_list,
    columns=term_list
)

# object to be stored
store = {"annotation": df_filter_annotation}

# make up train & test dataset masks for each fold
train_dataset = [dict() for _ in range(5)]
test_dataset = [dict() for _ in range(5)]
for fold in range(5):
    test_proteins = protein_list[fold::5]
    train_proteins = list(set(protein_list) - set(test_proteins))
    for term in filter_annotation:
        test_dataset[fold][term] = test_proteins
        train_dataset[fold][term] = train_proteins

# transform train & test masks to sparse matrices
store["mask"] = [dict() for _ in range(5)]

for fold in range(5):
    # Build sparse matrix for train mask
    train_rows, train_cols, train_data = [], [], []
    for term_idx, term in enumerate(term_list):
        proteins = train_dataset[fold][term]
        for protein in proteins:
            if protein in protein_to_idx:  # Ensure protein exists in mapping
                protein_idx = protein_to_idx[protein]
                train_rows.append(protein_idx)
                train_cols.append(term_idx)
                train_data.append(1)

    sparse_train = sparse.csr_matrix((train_data, (train_rows, train_cols)),
                                     shape=(len(protein_list), len(term_list)),
                                     dtype=np.int8)

    # Build sparse matrix for test mask
    test_rows, test_cols, test_data = [], [], []
    for term_idx, term in enumerate(term_list):
        proteins = test_dataset[fold][term]
        for protein in proteins:
            if protein in protein_to_idx:  # Ensure protein exists in mapping
                protein_idx = protein_to_idx[protein]
                test_rows.append(protein_idx)
                test_cols.append(term_idx)
                test_data.append(1)

    sparse_test = sparse.csr_matrix((test_data, (test_rows, test_cols)),
                                    shape=(len(protein_list), len(term_list)),
                                    dtype=np.int8)

    # Convert to DataFrame with sparse format
    df_train_dataset = pd.DataFrame.sparse.from_spmatrix(
        sparse_train,
        index=protein_list,
        columns=term_list
    )

    df_test_dataset = pd.DataFrame.sparse.from_spmatrix(
        sparse_test,
        index=protein_list,
        columns=term_list
    )

    store["mask"][fold]["train"] = df_train_dataset
    store["mask"][fold]["test"] = df_test_dataset

# write to pickle file
with open(config["dataset"], "wb") as fp:
    pickle.dump(store, fp)