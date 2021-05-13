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

def get_major_writers_pickle():
    states = ["Connecticut", "Georgia", "Massachussetts", "Pennsylvania", "Maryland", "NewYork", "Virginia"]
    states_writers = [cn_writers, ga_writers, ma_writers, pa_writers, md_writers, ny_writers, va_writers]

    middle_states = ["Pennsylvania", "NewYork"]
    middle_writers = [pa_writers, ny_writers]

    southern_states = ["Maryland", "Virginia", "Georgia"]
    southern_writers = [md_writers, va_writers, ga_writers]

    ne_states = ["Connecticut", "Massachussetts"]
    ne_writers = [cn_writers, ma_writers]


    major_authors = []
    for i in range(len(states)):
        state = states[i]
        file_name = state + ".pkl"
        csv_name = ".\\csv\\" + state + ".csv"

        with open(file_name, "rb") as handle:
            state_df = pickle.load(handle)

        state_table = pd.read_csv(csv_name)
        state_authors = state_table['Author']

        for j in range(len(state_authors)):
            writers_list = states_writers[i]

            if state_authors[j] in writers_list:
                print(state_authors[j])
                major_authors.append(state_df[j])

    with open("MajorWriters.pkl", "wb") as handle:
        pickle.dump(major_authors, handle)


    middle_authors = []
    for i in range(len(middle_states)):
        state = middle_states[i]
        file_name = state + ".pkl"
        csv_name = ".\\csv\\" + state + ".csv"

        with open(file_name, "rb") as handle:
            state_df = pickle.load(handle)

        state_table = pd.read_csv(csv_name)
        state_authors = state_table['Author']

        for j in range(len(state_authors)):
            writers_list = middle_writers[i]

            if state_authors[j] in writers_list:
                print(state_authors[j])
                middle_authors.append(state_df[j])

    with open("MiddleWriters.pkl", "wb") as handle:
        pickle.dump(middle_authors, handle)


    southern_authors = []
    for i in range(len(southern_states)):
        state = southern_states[i]
        file_name = state + ".pkl"
        csv_name = ".\\csv\\" + state + ".csv"

        with open(file_name, "rb") as handle:
            state_df = pickle.load(handle)

        state_table = pd.read_csv(csv_name)
        state_authors = state_table['Author']

        for j in range(len(state_authors)):
            writers_list = southern_writers[i]

            if state_authors[j] in writers_list:
                print(state_authors[j])
                southern_authors.append(state_df[j])

    with open("SouthernWriters.pkl", "wb") as handle:
        pickle.dump(southern_authors, handle)


    ne_authors = []
    for i in range(len(ne_states)):
        state = ne_states[i]
        file_name = state + ".pkl"
        csv_name = ".\\csv\\" + state + ".csv"

        with open(file_name, "rb") as handle:
            state_df = pickle.load(handle)

        state_table = pd.read_csv(csv_name)
        state_authors = state_table['Author']

        for j in range(len(state_authors)):
            writers_list = ne_writers[i]

            if state_authors[j] in writers_list:
                print(state_authors[j])
                ne_authors.append(state_df[j])

    with open("NEWriters.pkl", "wb") as handle:
        pickle.dump(ne_authors, handle)
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

def test_model(dict, texts, corpus):
    model_list, coherence_values = compute_coherence_values(state_dict=dict, corpus=corpus,
                                                            texts=texts, start=3,
                                                            limit=30, step=3)
    limit = 30
    start = 3
    step = 3
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


def topic_model_writers():
    """
    with open("NEWriters.pkl", "rb") as handle:
        ne_df = pickle.load(handle)

    ne_words = [nltk.word_tokenize(sent) for sent in ne_df]
    ne_dict = corpora.Dictionary(ne_words)
    ne_text = ne_words
    ne_corpus = [ne_dict.doc2bow(text) for text in ne_text]

    #test_model(ne_dict, ne_text, ne_corpus)

    ne_model = gensim.models.ldamodel.LdaModel(corpus=ne_corpus,
                                                     id2word=ne_dict,
                                                     num_topics=18,
                                                     random_state=100,
                                                     update_every=1,
                                                     chunksize=100,
                                                     passes=10,
                                                     alpha='auto',
                                                     per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(ne_model, ne_corpus, ne_dict)
    pyLDAvis.save_html(vis, fileobj="NEWritersVisualization.html")
    """
    """
    with open("SouthernWriters.pkl", "rb") as handle:
        southern_df = pickle.load(handle)

    southern_words = [nltk.word_tokenize(sent) for sent in southern_df]
    southern_dict = corpora.Dictionary(southern_words)
    southern_text = southern_words
    southern_corpus = [southern_dict.doc2bow(text) for text in southern_text]

    #test_model(southern_dict, southern_text, southern_corpus)


    southern_model = gensim.models.ldamodel.LdaModel(corpus=southern_corpus,
                                                   id2word=southern_dict,
                                                   num_topics=18,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(southern_model, southern_corpus, southern_dict)
    pyLDAvis.save_html(vis, fileobj="SouthernWritersVisualization.html")
    """
    """
    with open("MiddleWriters.pkl", "rb") as handle:
        middle_df = pickle.load(handle)

    middle_words = [nltk.word_tokenize(sent) for sent in middle_df]
    middle_dict = corpora.Dictionary(middle_words)
    middle_text = middle_words
    middle_corpus = [middle_dict.doc2bow(text) for text in middle_text]

    middle_model = gensim.models.ldamodel.LdaModel(corpus=middle_corpus,
                                                    id2word=middle_dict,
                                                    num_topics=12,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(middle_model, middle_corpus, middle_dict)
    pyLDAvis.save_html(vis, fileobj="MiddleWritersVisualization.html")

"""

    with open("MajorWriters.pkl", "rb") as handle:
        writers_df = pickle.load(handle)

    writers_words = [nltk.word_tokenize(sent) for sent in writers_df]
    writers_dict = corpora.Dictionary(writers_words)
    writers_texts = writers_words
    writers_corpus = [writers_dict.doc2bow(text) for text in writers_texts]

    #test_model(writers_dict, writers_texts, writers_corpus)

    writers_model = gensim.models.ldamodel.LdaModel(corpus=writers_corpus,
                                                    id2word=writers_dict,
                                                    num_topics=6,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(writers_model, writers_corpus, writers_dict)
    pyLDAvis.save_html(vis, fileobj="WritersVisualization.html")

def elites_and_popular():
    with open("Elites.pkl", "rb") as handle:
        elites_df = pickle.load(handle)

    elites_words = [nltk.word_tokenize(sent) for sent in elites_df]
    elites_dict = corpora.Dictionary(elites_words)
    elites_texts = elites_words
    elites_corpus = [elites_dict.doc2bow(text) for text in elites_texts]

    with open("Popular.pkl", "rb") as handle:
        popular_df = pickle.load(handle)

    popular_words = [nltk.word_tokenize(sent) for sent in popular_df]
    popular_dict = corpora.Dictionary(popular_words)
    popular_texts = popular_words
    popular_corpus = [popular_dict.doc2bow(text) for text in popular_texts]

    elites_model = gensim.models.ldamodel.LdaModel(corpus=elites_corpus,
                                                id2word=elites_dict,
                                                num_topics=8,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(elites_model, elites_corpus,elites_dict)
    pyLDAvis.save_html(vis, fileobj="ElitesVisualization.html")

    popular_model = gensim.models.ldamodel.LdaModel(corpus=popular_corpus,
                                                   id2word=popular_dict,
                                                   num_topics=20,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(popular_model, popular_corpus, popular_dict)
    pyLDAvis.save_html(vis, fileobj="PopularVisualization.html")

    """
    model_list, coherence_values = compute_coherence_values(state_dict=elites_dict, corpus=elites_corpus, texts=elites_texts, start=12,
                                                            limit=45, step=3)
    limit = 45
    start = 12
    step = 3
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()"""

if __name__ == '__main__':
    freeze_support()

    #get_major_writers_pickle()
    topic_model_writers()

    """
    with open("Virginia.pkl", "rb") as handle:
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


    #Building Topic Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=state_dict,
                                                num_topics=18,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=20,
                                                alpha='auto',
                                                per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, state_dict)
    pyLDAvis.save_html(vis, fileobj="VirginiaVisualization.html")


    #test_model(state_dict, texts, corpus)
"""



