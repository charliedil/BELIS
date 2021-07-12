import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import  cross_validate, StratifiedKFold
import numpy as np


def k_mean_cluster(docs):
    embeddings = []
    for doc in docs:
        for j in range(len(doc.user_data["subword_embeddings"])):
            for i in range(len(doc.user_data["ents"][j])):
                if(len(doc.user_data["ents"][j])!=1):
                    embeddings.append(doc.user_data["subword_embeddings"][j][i])
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    return kmeans.labels_

def nearest_centroid_classifier(docs):
    X = []
    y = []

    for doc in docs:

        for i in range(len(doc.user_data["subword_embeddings"])):
            for j in range(len(doc.user_data["subword_embeddings"][i])):
                X.append(doc.user_data["subword_embeddings"][i][j])

                y.append(doc.user_data["ents"][i][j].split("-")[1])


    # clf = NearestCentroid()
    # cv = StratifiedKFold(shuffle=True) #default is 5 splits
    # scoring = ['precision_macro','recall_macro','f1_macro']
    # scores = cross_validate(clf, X, y, cv=cv,scoring=scoring )
    # print(scores)
    cv = StratifiedKFold(shuffle=True, n_splits=5)
    iter = 1
    for train, test in cv.split(X,y):
        print("CROSS EVAL ITER: "+str(iter))
        iter+=1
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for i in range(len(train)):
            X_train.append(X[train[i]])
            y_train.append(y[train[i]])
        for i in range(len(test)):
            X_test.append(X[test[i]])
            y_test.append(y[test[i]])
        kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train)
        label_dist = {}
        label_mapping = {}
        labels = {"Other":0, "Drug":0, "Reason":0, "Route":0, "Form":0, "ADE":0, "Duration":0, "Strength":0, "Dosage":0, "Frequency":0}
        for i in range(len(kmeans.labels_)):
            labels[y_train[i]] +=1
            if kmeans.labels_[i] not in label_dist:
                label_dist[kmeans.labels_[i]] = {}
            if y_train[i] not in label_dist[kmeans.labels_[i]]:
                label_dist[kmeans.labels_[i]][y_train[i]] = 0
            label_dist[kmeans.labels_[i]][y_train[i]] +=1
        labels_sorted = dict(sorted(labels.items(), key=lambda item: item[1], reverse=True))
        for label in labels_sorted:
            max_centroid = -1
            max_value = 0
            for i in range(10):
                if label in label_dist[i] and label_dist[i][label]> max_value and i not in label_mapping:
                    max_value = label_dist[i][label]
                    max_centroid = i
            label_mapping[max_centroid] = label
        false_positives = {}
        true_positives = {}
        false_negatives = {}
        true_negatives = {}









