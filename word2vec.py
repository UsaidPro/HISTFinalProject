import pickle
import multiprocessing
from gensim.models import Word2Vec
import nltk
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.vq import vq

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random


with open("Maryland.pkl", "rb") as handle:
    maryland_df = pickle.load(handle)

maryland_df = [nltk.word_tokenize(sent) for sent in maryland_df]

cores = multiprocessing.cpu_count() - 1
w2v_model = Word2Vec(maryland_df,
                     window=10,
                     min_count=2,
                     workers=cores)

print(w2v_model.wv.similarity(w1="tyranny", w2="authority"))

#######################################################

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return labels, x_vals, y_vals, vectors


word_labels, x_vals, y_vals, vectors = reduce_dimensions(w2v_model)
pca = PCA(2)
df = pca.fit_transform(vectors)
kmeans = KMeans(n_clusters=12)
label = kmeans.fit_predict(df)
print(label)
print(word_labels)

# Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

# plotting the results:

plt.scatter(x_vals, y_vals)

for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)

plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.legend()

#
# Label randomly subsampled 25 data points
#


plt.show()

coord = np.array([x_vals, y_vals, label])
sorted_coord = coord[:,coord[-1].argsort()]




"""

#########################################
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(w2v_model)


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 50)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()


plot_function = plot_with_matplotlib


plot_function(x_vals, y_vals, labels)"""

