import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

#input_term = """21454

#Lycurgus III.

#A sermon to the bucks and hinds of America. :
#(Two lines from Genesis] : In imitation of a pamphlet
#entitled \"Sermons to asses-- and another to doctors of
#divinity.\" -- Philadelphia: : Printed for the author.,
#M.DCC.LXXXVIII. [1788] -- iv, 31, [1] p.; 19 cm.
#(12mo)
#ee ee

#true fons of Gol were fo by this fupernatural regeneration, and b
#born again of the fpirit of God—and that the ceremonies of reli
#were not effential, but that there would be oftentimes diffenters from
#: the firit and the former eftablifhed forms of religion ; fur which rea=

#fon he adopted Jofeph’s two fons, who were fuch diffenters. He faw
#that Jofeph’s heavenly father had bleffed him with an oflepring above
#thofe of the natural birth, and that thofe of the heavenly birth thould
#prevail above thofe of the natural feed, to the inhabiting the everlult-
#ing hills of the favage world in the latter days."""
file = open('../lycurgusOCR.txt', 'r', encoding='utf-8')
file_contents = ''
for line in file:
    if(len(line) < 15):
        continue
    file_contents += line
suggestions = sym_spell.lookup_compound(file_contents, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion)