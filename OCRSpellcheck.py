import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

file = open('../lycurgusOCR.txt', 'r', encoding='utf-8')
file_contents = ''
for line in file:
    if(len(line) < 15):
        continue
    file_contents += line
suggestions = sym_spell.lookup_compound(file_contents, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion)