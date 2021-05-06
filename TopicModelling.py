#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-2/

import pickle
import gensim
import nltk
from gensim import corpora ## Version 3.8.3
from gensim.models import wrappers
import pyLDAvis ## Version 2.1.2
import pyLDAvis.gensim
import os

with open("Pennsylvania.pkl", "rb") as handle:
    maryland_df = pickle.load(handle)

maryland_words = [nltk.word_tokenize(sent) for sent in maryland_df]

# Create dict
maryland_dict = corpora.Dictionary(maryland_words)

# Create corpus
texts = maryland_words

# Term document frequency
corpus = [maryland_dict.doc2bow(text) for text in texts]

# Human Readable form of the Above
read_corpus = [[(maryland_dict[id], freq) for id, freq in cp] for cp in corpus]


#Building Topic Model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=maryland_dict,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)


print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Visualize Topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, maryland_dict)
pyLDAvis.save_html(vis, 'Pennsylvania_Visualization.html')