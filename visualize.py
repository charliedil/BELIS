import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from clustering_algos import k_mean_cluster
import umap

def get_entity_embeddings_and_labels(doc): #exclude other
    embeddings = []
    entity_labels = []
    prev = False
    temp_entity_subword_embeddings = []
    temp_label = ""

    for j in range(len(doc.user_data["subword_embeddings"])):
        if len(doc.user_data["ents"][j]) > 0:
            if doc.user_data["ents"][j][0].split("-")[1] == "Other":
                if prev==True:
                    embeddings.append(np.mean(temp_entity_subword_embeddings, axis=0))
                    entity_labels.append(temp_label)
                    temp_entity_subword_embeddings = []
                    temp_label = ""
                    prev=False
            else:
                if prev==True:
                    if temp_label == doc.user_data["ents"][j][0].split("-")[1]:
                        for subword_embedding in doc.user_data["subword_embeddings"][j]:
                            temp_entity_subword_embeddings.append(subword_embedding)
                    else:
                        embeddings.append(np.mean(temp_entity_subword_embeddings, axis=0))
                        entity_labels.append(temp_label)
                        temp_entity_subword_embeddings = []
                        for subword_embedding in doc.user_data["subword_embeddings"][j]:
                            temp_entity_subword_embeddings.append(subword_embedding)
                        temp_label = doc.user_data["ents"][j][0].split("-")[1]
                else:
                    for subword_embedding in doc.user_data["subword_embeddings"][j]:
                        temp_entity_subword_embeddings.append(subword_embedding)
                    temp_label = doc.user_data["ents"][j][0].split("-")[1]
                    prev=True
    return embeddings, entity_labels



def get_word_embeddings_and_labels(doc):
    embeddings = []
    entity_labels = []

    for j in range(len(doc.user_data["subword_embeddings"])):
        if len(doc.user_data["ents"][j]) > 0:

            embeddings.append(np.mean(doc.user_data["subword_embeddings"][j], axis=0))
            entity_labels.append(doc.user_data["ents"][j][0].split("-")[1])
            assert (len(np.mean(doc.user_data["subword_embeddings"][j], axis=0)) ==768)


    return embeddings, entity_labels

def get_subword_embeddings_and_labels(doc):
    embeddings = []
    entity_labels = []
    for j in range(len(doc.user_data["subword_embeddings"])):
        for i in range(len(doc.user_data["ents"][j])):
            if(len(doc.user_data["ents"][j])!=1): #Pretty sure i fixed it so this is irrelevant now
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
    mapping["Other"] = []
    i=0
    for key in entity_labels:
        mapping[key].append(embeddings[i])
        i+=1
    return mapping

# def draw(path):
#     doc_bin = DocBin().from_disk(path)
#     vocab = Vocab().from_disk("C:/Users/nehav/Desktop/n2c2_100035_vocab.spacy")
#     docs = list(doc_bin.get_docs(vocab))
#     embeddings, labels = get_subword_embeddings_and_labels(docs[0])
#     k_cluster_labels = k_mean_cluster(docs[0])
#     entity_label_to_embedding_mapping = map_embedding_to_entity(embeddings, labels)
#     for key in entity_label_to_embedding_mapping:
#         print(key+": "+str(len(entity_label_to_embedding_mapping[key])))
#     sns.set(style='white', rc={'figure.figsize':(12,8)})
#
#  #   print(data)
#     i=0
#     fit = umap.UMAP(n_neighbors=10)
#     color_map = {"Drug":"palevioletred", "Reason":"plum", "Route":"mediumpurple","Form":"skyblue","ADE":"mediumseagreen", "Duration":"blue", "Strength":"orange","Dosage":"brown","Frequency":"gray"}#,"Other":"yellow"}
#     color_map2 = {0:"palevioletred", 1:"plum", 2:"mediumpurple",3:"skyblue",4:"mediumseagreen", 5:"blue", 6:"orange",7:"brown",8:"gray"}#, 9:"yellow"}
#     colors = []
#     colors2 = []
#     filtered_embeddings = []
#     for i in range(len(labels)):
#         if labels[i] in color_map:
#             colors.append(color_map[labels[i]])
#             filtered_embeddings.append(embeddings[i])
#             colors2.append(color_map2[k_cluster_labels[i]])
#
#     u = fit.fit_transform(filtered_embeddings)
#     plt.scatter(u[:, 0], u[:, 1], c=colors)
#
#     plt.show()
#
#     plt.scatter(u[:, 0], u[:, 1], c=colors2)
#     plt.show()
def reorder(embeddings, labels, bg_label):
    ordered_embeddings = []
    ordered_labels = []
    for i in range(len(labels)):
        if labels[i]==bg_label:
            ordered_labels.append(labels[i])
            ordered_embeddings.append(embeddings[i])
    for i in range(len(labels)):
        if labels[i]!=bg_label:
            ordered_labels.append(labels[i])
            ordered_embeddings.append(embeddings[i])
    assert(len(ordered_embeddings)==len(embeddings))
    assert(len(ordered_labels)==len(labels))
    return ordered_embeddings, ordered_labels

def draw_word_level(path):
    doc_bin = DocBin().from_disk(path)
    vocab = Vocab().from_disk("C:/Users/nehav/Desktop/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    embeddings, labels = get_entity_embeddings_and_labels(docs[0]) ##depending on which level of embeddings you want, change this!!
    ordered_embeddings, ordered_labels = reorder(embeddings, labels, "Other")
    k_cluster_labels = k_mean_cluster(ordered_embeddings)
    entity_label_to_embedding_mapping = map_embedding_to_entity(ordered_embeddings, ordered_labels)
    for key in entity_label_to_embedding_mapping:
        print(key+": "+str(len(entity_label_to_embedding_mapping[key])))
    sns.set(style='white', rc={'figure.figsize':(12,8)})

 #   print(data)
    i=0
    fit = umap.UMAP(n_neighbors=10)
    color_map = {"Drug":"palevioletred", "Reason":"plum", "Route":"mediumpurple","Form":"skyblue","ADE":"mediumseagreen", "Duration":"blue", "Strength":"orange","Dosage":"brown","Frequency":"yellow","Other":"lightgray"}
    color_map2 = {0:"palevioletred", 1:"plum", 2:"mediumpurple",3:"skyblue",4:"mediumseagreen", 5:"blue", 6:"orange",7:"brown",8:"gray"}#, 9:"yellow"}
    colors = []
    opacity = []
    colors2 = []
    filtered_embeddings = []
    for i in range(len(ordered_labels)):
        if ordered_labels[i] in color_map:
            colors.append(color_map[ordered_labels[i]])
            filtered_embeddings.append(ordered_embeddings[i])
            colors2.append(color_map2[k_cluster_labels[i]])
            if ordered_labels[i] == "Other":
                opacity.append(.25)
            else:
                opacity.append(1.0)

    u = fit.fit_transform(filtered_embeddings)
    plt.scatter(u[:, 0], u[:, 1], c=colors, alpha=opacity)

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Drug',
                          markerfacecolor='palevioletred', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Reason',
                              markerfacecolor='plum', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Route',
                              markerfacecolor='mediumpurple', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Form',
                              markerfacecolor='skyblue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='ADE',
                              markerfacecolor='mediumseagreen', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Duration',
                              markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Strength',
                              markerfacecolor='orange', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Dosage',
                              markerfacecolor='brown', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Frequency',
                              markerfacecolor='yellow', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Other',
                              markerfacecolor='lightgray', markersize=15)
                       ]
    ax.legend(handles=legend_elements,bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    plt.scatter(u[:, 0], u[:, 1], c=colors2)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster 1',
                              markerfacecolor='palevioletred', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 2',
                              markerfacecolor='plum', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 3',
                              markerfacecolor='mediumpurple', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 4',
                              markerfacecolor='skyblue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 5',
                              markerfacecolor='mediumseagreen', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 6',
                              markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 7',
                              markerfacecolor='orange', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 8',
                              markerfacecolor='brown', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 9',
                              markerfacecolor='gray', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 10',
                              markerfacecolor='yellow', markersize=15)
                       ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
