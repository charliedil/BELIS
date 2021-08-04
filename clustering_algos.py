import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import  cross_validate, StratifiedKFold
import numpy as np

def get_word_embeddings_and_labels(doc):
    embeddings = []
    entity_labels = []

    for j in range(len(doc.user_data["subword_embeddings"])):
        if len(doc.user_data["ents"][j]) > 0:

            embeddings.append(np.mean(doc.user_data["subword_embeddings"][j], axis=0))
            entity_labels.append(doc.user_data["ents"][j][0].split("-")[1])
            assert (len(np.mean(doc.user_data["subword_embeddings"][j], axis=0)) ==768)


    return embeddings, entity_labels

def k_mean_cluster(embeddings):
    kmeans = KMeans(n_clusters=10 , random_state=0).fit(embeddings)
    return kmeans.labels_

def nearest_centroid_classifier(docs):
    X = []
    y = []

    for doc in docs:
        doc_embeddings, doc_labels = get_word_embeddings_and_labels(doc)
        for i in range(len(doc_embeddings)):
            X.append(doc_embeddings[i])
            y.append(doc_labels[i])


    # clf = NearestCentroid()
    # cv = StratifiedKFold(shuffle=True) #default is 5 splits
    # scoring = ['precision_macro','recall_macro','f1_macro']
    # scores = cross_validate(clf, X, y, cv=cv,scoring=scoring )
    # print(scores)
    cv = StratifiedKFold(shuffle=True, n_splits=2)
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
        print(labels_sorted)
        for label in labels_sorted:
            max_centroid = -1
            max_value = 0
            for i in range(10):
                if label in label_dist[i] and label_dist[i][label]>= max_value and i not in label_mapping:
                    max_value = label_dist[i][label]
                    max_centroid = i
            if max_centroid!=-1:
                label_mapping[max_centroid] = label
        for i in range(10):
            if i not in label_mapping:
                for label in labels_sorted:
                    if label not in list(label_mapping.values()):
                        label_mapping[i] = label
                        break
        false_positives = {}
        true_positives = {}
        false_negatives = {}
        true_negatives = {}
        for label in labels:
            false_positives[label] = 0
            false_negatives[label] = 0
            true_negatives[label] = 0
            true_positives[label] = 0
        output_labels = kmeans.predict(X_test)

        for i in range(len(output_labels)):
            if label_mapping[output_labels[i]] == y_test[i]:
                true_positives[y_test[i]] +=1
                for label in labels:
                    if label != y_test[i]:
                        true_negatives[label] +=1
            else:
                false_negatives[y_test[i]]+=1
                false_positives[label_mapping[output_labels[i]]] +=1
                for label in labels:
                    if label!= y_test[i] and label!=label_mapping[output_labels[i]]:
                        true_negatives[label]+=1
        precision_temp = 0
        recall_temp = 0
        f1_temp = 0
        for label in labels:
            print("Class: "+label)
            print("Precision: "+str(true_positives[label]/(false_positives[label]+true_positives[label])))
            print("Recall: "+str(true_positives[label]/(false_negatives[label]+true_positives[label])))
            print("F1: "+str(true_positives[label]/(true_positives[label]+.5*false_positives[label]+.5*false_negatives[label])))
            precision_temp += true_positives[label]/(false_positives[label]+true_positives[label])
            recall_temp += true_positives[label]/(false_negatives[label]+true_positives[label])
            f1_temp += true_positives[label]/(true_positives[label]+.5*false_positives[label]+.5*false_negatives[label])
        print("OVERALL -------------------------")
        print("Precision: "+str(precision_temp/10))
        print("Recall: "+str(recall_temp/10))
        print("F1: "+str(f1_temp/10))












