import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from sklearn.cluster import KMeans
import numpy as np

def k_mean_cluster(doc):
    embeddings = []
    for j in range(len(doc.user_data["subword_embeddings"])):
        for i in range(len(doc.user_data["ents"][j])):
            if(len(doc.user_data["ents"][j])!=1):
                embeddings.append(doc.user_data["subword_embeddings"][j][i])
    kmeans = KMeans(n_clusters=6, random_state=0).fit(embeddings)
    return kmeans.labels_