import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
import umap

def create_mapping(doc):
    mapping = {}
    mapping["Drug"] = []
    mapping["Dosage"] = []
    mapping["Duration"] = []
    mapping["Frequency"] = []
    mapping["ADE"] = []
    mapping["Form"] = []
    mapping["Reason"] = []
    mapping["Strength"] = []
    mapping["Route"] = []
    for j in range(len(doc.user_data["subword_embeddings"])):
        for i in range(len(doc.user_data["ents"][j])):
            if(len(doc.user_data["ents"][j])!=1):
                mapping[doc.user_data["ents"][j][i].split("-")[1]].append(doc.user_data["subword_embeddings"][j][i])
    return mapping


def draw(path):
    doc_bin = DocBin().from_disk(path)
    vocab = Vocab().from_disk("C:/Users/nehav/Desktop/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    entity_embeddings = create_mapping(docs[0])
    for key in entity_embeddings:
        print(key+": "+str(len(entity_embeddings[key])))
    sns.set(style='white', rc={'figure.figsize':(12,8)})

 #   print(data)
    i=0
    fit = umap.UMAP(n_neighbors=5)

    colors = ["palevioletred", "plum", "mediumpurple", "cornflowerblue", "skyblue", "mediumseagreen", "navajowheat", "darksalmon", "silver"]
    for label in ["Duration","Drug", "Reason", "ADE", "Form", "Route"]:
        print(label)

        u = fit.fit_transform(entity_embeddings[label])
        plt.scatter(u[:, 0], u[:, 1], color=colors[i], label = label)
        i+=1

    plt.legend()
    plt.show()