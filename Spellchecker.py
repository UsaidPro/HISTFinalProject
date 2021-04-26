from symspellpy import SymSpell
import pkg_resources
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import re
import multiprocessing

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


def preprocess(sentences):
    corrected_sentences = []
    for term in sentences:
        term = re.sub(r'[.?!,:;()\-\n\d]', ' ', term)
        tokens = [t.lower() for t in word_tokenize(term) if t not in stopwords.words('english')]

        wnl = WordNetLemmatizer()
        term = " ".join(wnl.lemmatize(t) for t in tokens)

        suggestions = sym_spell.lookup_compound(term, max_edit_distance=2)
        corrected = " ".join([str(elem) for elem in suggestions])
        corrected_sentences.append(corrected)
        print(corrected)

    return corrected_sentences

"""
conn_df = pd.read_csv("Connecticut.csv")
conn_sentences = conn_df['Quote']
conn_df['Quote'] = preprocess(conn_sentences)
print(conn_df)

sc_df = pd.read_csv("South Carolina.csv")
sc_sentences = sc_df['Quote']
sc_df['Quote'] = preprocess(sc_sentences)
print(sc_df)
"""

conn_df = pd.read_csv("Connecticut.csv")
conn_sentences = conn_df['Quote']

sc_df = pd.read_csv("South Carolina.csv")
sc_sentences = sc_df['Quote']

p1 = multiprocessing.Process(target=preprocess, args=(conn_sentences,))
p2 = multiprocessing.Process(target=preprocess, args=(conn_sentences,))

p1.start()
p2.start()

results = []
results.append(p1)
results.append(p2)
