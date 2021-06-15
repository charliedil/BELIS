import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import  cross_validate, StratifiedKFold
import numpy as np


def k_mean_cluster(docs):
    embeddings = []
    for doc in docbin:
        for j in range(len(doc.user_data["subword_embeddings"])):
            for i in range(len(doc.user_data["ents"][j])):
                if(len(doc.user_data["ents"][j])!=1):
                    embeddings.append(doc.user_data["subword_embeddings"][j][i])
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    return kmeans.labels_

def nearest_centroid_classifier(docs):
    X = []
    y = []
    subword_labels = []
    for doc in docs:
        for i in range(len(doc.user_data["subword_embeddings"])):
            for j in range(len(doc.user_data["subword_embeddings"][i])):
                X.append(doc.user_data["subword_embeddings"][i][j])
                subword_labels.append(doc.user_data["ents"][i][j].split("-")[1])
    label_mapping = {"Drug":1, "Reason":2, "Route":3,"Form":4,"ADE":5, "Duration":6, "Strength":7,"Dosage":8,"Frequency":9,"Other":0}
    for label in subword_labels:
        y.append(label_mapping[label])
    clf = NearestCentroid()
    cv = StratifiedKFold(shuffle=True) #default is 5 splits
    scoring = ['precision_macro','recall_macro','f1_macro']
    scores = cross_validate(clf, X, y, cv=cv,scoring=scoring )
    print(scores)