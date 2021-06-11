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
from kmeans_clustering import k_mean_cluster
import umap

def get_embeddings_and_labels(doc):
    # mapping = {}
    # mapping["Drug"] = []
    # mapping["Dosage"] = []
    # mapping["Duration"] = []
    # mapping["Frequency"] = []
    # mapping["ADE"] = []
    # mapping["Form"] = []
    # mapping["Reason"] = []
    # mapping["Strength"] = []
    # mapping["Route"] = []
    embeddings = []
    entity_labels = []
    for j in range(len(doc.user_data["subword_embeddings"])):
        for i in range(len(doc.user_data["ents"][j])):
            if(len(doc.user_data["ents"][j])!=1):
                entity_labels.append(doc.user_data["ents"][j][i].split("-")[1])
                embeddings.append(doc.user_data["subword_embeddings"][j][i])
    return embeddings, entity_labels

def map_embedding_to_entity(embeddings, entity_labels):
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
    i=0
    for key in entity_labels:
        mapping[key].append(embeddings[i])
        i+=1
    return mapping

def map_kcluster_label_to_embedding(embeddings, cluster_labels):
    mapping = {}
    mapping[0] = []
    mapping[1] = []
    mapping[2] = []
    mapping[3] = []
    mapping[4] = []
    mapping[5] = []
    mapping[6] = []
    mapping[7] = []
    mapping[8] = []
    i = 0
    for key in cluster_labels:
        mapping[key].append(embeddings[i])
        i += 1
    return mapping

def draw(path):
    doc_bin = DocBin().from_disk(path)
    vocab = Vocab().from_disk("C:/Users/nehav/Desktop/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    embeddings, labels = get_embeddings_and_labels(docs[0])
    k_cluster_labels = k_mean_cluster(docs[0])
    entity_label_to_embedding_mapping = map_embedding_to_entity(embeddings, labels)
    for key in entity_label_to_embedding_mapping:
        print(key+": "+str(len(entity_label_to_embedding_mapping[key])))
    sns.set(style='white', rc={'figure.figsize':(12,8)})

 #   print(data)
    i=0
    fit = umap.UMAP(n_neighbors=150)
    color_map = {"Drug":"palevioletred", "Reason":"plum", "Route":"mediumpurple","Form":"skyblue","ADE":"mediumseagreen", "Duration":"blue", "Strength":"orange","Dosage":"brown","Frequency":"gray"}
    color_map2 = {0:"palevioletred", 1:"plum", 2:"mediumpurple",3:"skyblue",4:"mediumseagreen", 5:"blue", 6:"orange",7:"brown",8:"gray"}
    colors = []
    colors2 = []
    filtered_embeddings = []
    for i in range(len(labels)):
        if labels[i] in color_map:
            colors.append(color_map[labels[i]])
            filtered_embeddings.append(embeddings[i])
            colors2.append(color_map2[k_cluster_labels[i]])

    u = fit.fit_transform(filtered_embeddings)
    plt.scatter(u[:, 0], u[:, 1], c=colors)

    plt.show()

    plt.scatter(u[:, 0], u[:, 1], c=colors2)
    plt.show()
