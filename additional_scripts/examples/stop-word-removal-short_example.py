#https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936

import spacy
print('spaCy version: %s' % (spacy.__version__))
spacy_nlp = spacy.load('en_core_web_sm')

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#print(spacy_nlp.Defaults.stop_words)

#write stop-words to txt--------------
stopWords = spacy_nlp.Defaults.stop_words
f=open('stop-words_spacy.txt', 'w')
stopWords = map(lambda x:x+'\n', stopWords)
f.writelines(stopWords)
f.close()
#--------------
#print('Number of stop words: %d' % len(spacy_stopwords))
#print('First ten stop words: %s' % list (spacy_stopwords)[:10])

article = "In computing, stop words are words which are filtered out before or after processing of natural language data (text).[1] Though 'stop words' usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list. Some tools specifically avoid removing these stop words to support phrase search."

doc = spacy_nlp(article)
tokens = [token.text for token in doc if not token.is_stop]


#print('Original Article: %s' % (article))
#print()
#print(tokens)