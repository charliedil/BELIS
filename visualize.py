import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab

def draw(path):
    doc_bin = DocBin().from_disk(path)
    vocab = Vocab().from_disk("C:/Users/nehav/Desktop/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    embeddings = docs[0].user_data["subword_embeddings"]


    sns.set(style='white', rc={'figure.figsize':(12,8)})
    np.random.seed(42)
    data = np.random.rand(800, 4)
    fit = umap.UMAP()
    u = fit.fit_transform(data)
    plt.scatter(u[:,0], u[:,1], c=data)
    plt.show()