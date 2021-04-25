import pprint
import csv

from symspellpy import SymSpell
import pkg_resources
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import re
import pickle
import os

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

state = 'Maryland'
spellchecked_terms = []
if not os.path.exists(state + '.pkl'):
    df = pd.read_csv(state + '.csv')
    terms = df['Quote']

    count = 1
    for term in terms:
        term = re.sub(r'[.?!,:;()\-\n\d]', ' ', term)
        tokens = [t.lower() for t in word_tokenize(term) if t not in stopwords.words('english')]

        wnl = WordNetLemmatizer()
        term = " ".join(wnl.lemmatize(t) for t in tokens)

        suggestions = sym_spell.lookup_compound(term, max_edit_distance=2)
        corrected = " ".join([str(elem) for elem in suggestions])
        spellchecked_terms.append(corrected)
        count += 1
    with open(state + '.pkl', 'wb') as f:
        pickle.dump(spellchecked_terms, f)
else:
    with open(state + '.pkl', 'rb') as f:
        spellchecked_terms = pickle.load(f)

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in spellchecked_terms]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

pprint.pprint(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])

model = models.Word2Vec(sentences=processed_corpus)

for index, word in enumerate(model.wv.index_to_key):
    if index == 50:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


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


x_vals, y_vals, labels = reduce_dimensions(model)


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
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()

plot_with_matplotlib(x_vals, y_vals, labels)