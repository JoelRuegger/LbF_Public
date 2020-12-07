import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("Apples and oranges are similar. Boots and hippos aren't.")

#replace all words by their lemma, if no lemma is found, the original word is returned
tokens = [token.lemma_ for token in doc]
print(tokens)

#alternatively
# tokens = []
# for token in doc:
#     tokens.append(token.lemma_)
#print (tokens)
