from symspellpy import SymSpell
import pkg_resources
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import re


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

df = pd.read_csv("Connecticut.csv")

terms = df['Quote']
spellchecked_terms = []

count = 1
for term in terms:
    term = re.sub(r'[.?!,:;()\-\n\d]', ' ', term)
    tokens = [t.lower() for t in word_tokenize(term) if t not in stopwords.words('english')]

    wnl = WordNetLemmatizer()
    term = " ".join(wnl.lemmatize(t) for t in tokens)

    suggestions = sym_spell.lookup_compound(term, max_edit_distance=2)
    corrected = " ".join([str(elem) for elem in suggestions])
    spellchecked_terms.append(corrected)
    print(corrected)
    count += 1

df['Quote'] = spellchecked_terms
print(df)
