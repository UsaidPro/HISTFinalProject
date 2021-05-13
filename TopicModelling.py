#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-2/

import pickle
import gensim
import nltk
from gensim import corpora ## Version 3.8.3
from gensim.models import wrappers, CoherenceModel
import pyLDAvis ## Version 2.1.2
import pyLDAvis.gensim
import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


### Major Writers ###
cn_writers = ["Benjamin Gale"]
ga_writers = ["A Georgian"]
ma_writers = ["Vox Populi", "John De Witt", "William Symmes Jr.", "Agrippa", "A Federalist", "One of the Common People", "Candidus",
              "Cornelius", "Poplicola", "Helvidius Priscus", "The Republican Federalist", "Samuel", "Hampden (MA)", "The Yeomanry of Massachusetts",
              "An American", "Bostonian", "A Watchman", 'Consider Arms, Malachi Maynard, Samuel Field', "Phileleutheros",
              "Mercy Otis Warren", "John Quincy Adams", "Elbridge Gerry", "Hanno", "Adelos", "Portius", "Massachusettensis",
              "A Watchman"] ## MA anon Massachusetts Gazette 10/9/1787
pa_writers = ['James M\'Calmont, Robert Clark, Jacob Miley, Alexander Wright, John M\'Dowell, John Flenniken, James Allison, Theophilus Philips, John Gilchrist, Abraham Smith, Robert Whitehill, David Mitchel, John Piper, Samuel Dale, William Findley, James Barr',
              "Centinel", "An Old Whig", "A Democratic Federalist", "A Federal Republican", "John Humble", "An Officer of the Late Continental Army",
              "Philadelphiensis", "Alfred",
              'Nathaniel Breading, John Smilie, Richard Baird, Adam Orth, John A. Hanna, John Whitehill, John Harris, Robert Whitehill, John Reynolds, Jonathan Hoge, Nicholas Lutz, John Ludwig, Abraham Lincoln, John Bishop, Joseph Heister, Joseph Powel, James Martin, William Findley, John Baird, James Edgar, William Todd',
              "William Penn", "Deliberator", "Aristocrotis", "A Farmer (PA)", "None of the Well-Born Conspirators"]
md_writers = ['William Paca, Samuel Chase, John Francis Mercer, Jeremiah Chase, John Love, Charles Ridgley, Edward Cockey, Nathan Cromwell, Charles Ridgley (Wm.), Luther Martin, Benjamin Harrison, William Pinkney',
              "Luther Martin", "A Farmer", "A Farmer and Planter", "John Francis Mercer"]
ny_writers = ["Cincinnatus", "Brutus Junior", "Brutus", "Sidney", "Sydney", "Cato", "Federal Farmer", "A Countryman (I)",
              "A Countryman (II)", "Democritus", "A Son of Liberty", "An Observer", "George Clinton", "A Citizen",
              "Albany Anti-Federal Committee", "Expositor", "A Tenant", "A Plebeian", "A Republican", "Timoleon"]
va_writers = ["Richard Henry Lee", "Cato Uticencis", "Tamony", "George Mason", "Edmund Randolph", "James Monroe", "Brutus (VA)",
              "The Impartial Examiner", "The Society of Western Gentlemen Revise the Constitution", "Denatus", "William Grayson",
              "William Nelson Jr.", "Caleb Wallace", "A Virginia Planter", "Senex", "Republicus", "John Leland", "A Ploughman"]
######################

##### Saul Cornell Distinctions ###
elite_writers = ["Elbridge Gerry", "George Mason", "Luther Martin", "Mercy Otis Warren", "Cincinnatus", "Agrippa",
                 "John De Witt", "Richard Henry Lee", "The Republican Federalist", "William Grayson", "Cato Uticensis"]
popular_writers = ["An Old Whig", "Federal Farmer", "Brutus", "A Son of Liberty", "Centinel", "Philadelphiensis",
                   "Aristocrotis", "An Officer of the Late Continental Army", "A Plebeian", "Cato"]



"""
############Major Writers#################
with open("Pennsylvania.pkl", "rb") as handle:
    state_df = pickle.load(handle)

state_table = pd.read_csv("Pennsylvania.csv")
state_authors = state_table['Author']

authors_df = []

for i in range(len(state_authors)):
    if state_authors[i] in pa_writers:
        print(state_authors[i])
        authors_df.append(state_df[i])

author_words = [nltk.word_tokenize(sent) for sent in authors_df]

# Create dict
authors_dict = corpora.Dictionary(author_words)

# Create corpus
texts = author_words

# Term document frequency
corpus = [authors_dict.doc2bow(text) for text in texts]

# Human Readable form of the Above
read_corpus = [[(authors_dict[id], freq) for id, freq in cp] for cp in corpus]


#Building Topic Model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=authors_dict,
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
vis = pyLDAvis.gensim.prepare(lda_model, corpus, authors_dict)
pyLDAvis.save_html(vis, 'PA_Writers_Visualization.html')
"""
#######################################



###################States as a Whole#################################

def compute_coherence_values(state_dict, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=state_dict,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        model_list.append(lda_model)
        coherence_score = CoherenceModel(model=lda_model, dictionary=state_dict, texts=texts, coherence='c_v')
        score = coherence_score.get_coherence()
        coherence_values.append(score)

    return model_list, coherence_values


if __name__ == '__main__':
    freeze_support()

    with open("Maryland.pkl", "rb") as handle:
        state_df = pickle.load(handle)

    state_words = [nltk.word_tokenize(sent) for sent in state_df]

    # Create dict
    state_dict = corpora.Dictionary(state_words)

    # Create corpus
    texts = state_words

    # Term document frequency
    corpus = [state_dict.doc2bow(text) for text in texts]

    # Human Readable form of the Above
    read_corpus = [[(state_dict[id], freq) for id, freq in cp] for cp in corpus]

    """
    #Building Topic Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=state_dict,
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
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, state_dict)
    pyLDAvis.save_html(vis, 'Pennsylvania_Visualization.html')

    """

    model_list, coherence_values = compute_coherence_values(state_dict=state_dict, corpus=corpus, texts=texts, start=2,
                                                            limit=40, step=6)
    limit = 40
    start = 2
    step = 6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()





