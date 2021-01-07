# coding: utf8

# Source code: https://foad-moodle.ensai.fr/pluginfile.php/15490/mod_resource/content/4/tp1.pdf

import spacy

nlp = spacy.load("en_core_web_sm/en_core_web_sm-2.3.1")

# shows pipeline components (in order)
print(nlp.pipeline)

# ** Document **
text = "This is a $2 test sentence to test a U.K. spaCy pipline"
doc = nlp(text)  # -> apply pipeline on text

# iterate over tokens in doc
with open("data_out/token.txt", 'w') as tokf:
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, file=tokf)

# iterate over the entities detected
with open("data_out/entity.txt", 'w') as entf:
    for entity in doc.ents:
        print(entity.text, entity.start_char, entity.end_char, entity.label_, file=entf)

# visualisation of the dependency parse
spacy.displacy.render(doc, style="dep")

# ** Vocab **
with open("data_out/lexeme.txt", 'w') as lexf:
    for lexeme in nlp.vocab:       # -> iterate over the lexemes
        print(lexeme.text, lexeme.norm_, lexeme.lower_, lexeme.has_vector, file=lexf)
        # if lexeme.has_vector:    # commented because too verbose
        #    print(lexeme.vector)
