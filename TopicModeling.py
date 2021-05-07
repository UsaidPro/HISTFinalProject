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
from gensim.parsing.preprocessing import preprocess_string

# Create a set of frequent words
STOPWORDS = frozenset([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fifty', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once'
])

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

initial_state_list = ['Massachussetts', 'SouthCarolina', 'Virginia', 'Pennsylvania', 'NewHampshire', 'Connecticut', 'Georgia', 'Maryland']
#initial_state_list = ['Massachussetts', 'Virginia', 'Pennsylvania', 'Maryland']
state_list = []
#state = 'Maryland'
all_terms = []
all_labels = []
for state in initial_state_list:
    spellchecked_terms = []
    try:
        if not os.path.exists(state + '.pkl'):
            df = pd.read_csv(state + '.csv', delimiter=',')
            terms = df['Quote']

            count = 1
            for term in terms:
                term = re.sub(r'[.?!,:;()\-\n\d]', ' ', term)
                tokens = [t.lower() for t in word_tokenize(term) if t not in stopwords.words('english')]

                wnl = WordNetLemmatizer()
                term = " ".join(wnl.lemmatize(t) for t in tokens)
                #term_processed = " ".join(preprocess_string(term))

                suggestions = sym_spell.lookup_compound(term, max_edit_distance=2)
                for quote in suggestions:
                    quote_str = quote._term
                    spellchecked_terms.append(" ".join([i for i in quote_str.split(' ') if i not in stopwords.words('english') and i not in STOPWORDS]))
                    all_labels.append(" ".join([state for i in spellchecked_terms[-1].split(' ')]))
                #corrected = " ".join([str(elem) for elem in suggestions])
                #spellchecked_terms.append(corrected)
                count += 1
                #all_labels.append(" ".join([state for i in range(len(suggestions))]))
            with open(state + '.pkl', 'wb') as f:
                pickle.dump(spellchecked_terms, f)
        else:
            with open(state + '.pkl', 'rb') as f:
                spellchecked_terms = pickle.load(f)
            for term in spellchecked_terms:
                all_labels.append(" ".join([state for i in term.split()]))
        all_terms.extend(spellchecked_terms)
        state_list.append(state)
    except Exception:
        print('Issue with state ' + state)

# Lowercase each document, split it by white space and filter out stopwords
#texts = [[word for word in document.lower().split() if word not in stoplist]
#         for document in all_terms]
texts = []
labels = []
for i in range(len(all_terms)):
    word_list = []
    label_list = []
    document = all_terms[i]
    document_split = document.lower().split()
    labels_split = all_labels[i].split()
    for j in range(len(document_split)):
        if(document_split[j] not in STOPWORDS):
            word_list.append(document_split[j])
            label_list.append(labels_split[j])
    texts.append(word_list)
    labels.append(label_list)

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
processed_corpus = []
processed_corpus_labels = []
for i in range(len(texts)):
    processed_tokens = []
    processed_labels = []
    for j in range(len(texts[i])):
        if frequency[texts[i][j]] > 1:
            processed_tokens.append(texts[i][j] + '_' + str(state_list.index(labels[i][j])))
            processed_labels.append(labels[i][j])
    processed_corpus.append(processed_tokens)
    processed_corpus_labels.append(processed_labels)
#pprint.pprint(processed_corpus)

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
#print(dictionary)

#pprint.pprint(dictionary.token2id)


bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
#pprint.pprint(bow_corpus)

from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

model = models.Word2Vec(sentences=processed_corpus)

#for index, word in enumerate(model.wv.index_to_key):
#    if index == 1000:
#        break
#    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

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


import matplotlib.pyplot as plt
import matplotlib
import random

random.seed(0)

fig, ax = plt.subplots()
#plt.figure(figsize=(12, 12))
colors = [int(label.split('_')[1]) for label in labels]
color_list = ['red', 'blue', 'green', 'yellow', 'cyan', 'grey', 'purple', 'orange', 'pink']
for i in range(len(state_list)):
    x_vals_state = []
    y_vals_state = []
    colors_state = []
    for j in range(len(x_vals)):
        if colors[j] == i:
            x_vals_state.append(x_vals[j])
            y_vals_state.append(y_vals[j])
            colors_state.append(colors[j])
    ax.scatter(x_vals_state, y_vals_state, c=color_list[i], label=state_list[i])
#ax.scatter(x_vals, y_vals, c=colors)

#
# Label randomly subsampled 25 data points
#
indices = list(range(len(labels)))
selected_indices = random.sample(indices, 100)
for i in selected_indices:
    ax.annotate(labels[i].split('_')[0], (x_vals[i], y_vals[i]))
ax.legend(state_list, loc='lower left')
#ax.legend(('Massachussetts', 'Virginia', 'Pennsylvania', 'Maryland'), loc='lower left')
plt.show()