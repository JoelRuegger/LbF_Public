import glob
import spacy
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import csv

rel_path_to_trainingdata = 'partOfData/'    #'test_txt/'
file_ids = []
file_ratings = []
file_ratings_standardized = []
review_list = []
overall_dict = {} # {id: (rating, "full_cleaned_review", [(tok, lemma, pos)])}

######## variables for lexical resource ########
lexicon = []
lexiconAsCsv = 'WarrinerList2013.csv' # name of lexicon to use
wordIndex = 1 # index of wordtext in csv
valenceIndex = 2 # index of valence in csv
arousalIndex = 5 # index of arousal in csv
dominanceIndex = 8 # index of dominance in csv
lexicon_dict = {} # {'Word': ('V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum'), 'aardvark': ('6.26', '2.41', '4.27'), ... }

nlp = spacy.load('en_core_web_sm')

def read_first_line_txt(file):
    with open(file, 'r') as fd:
        first_line = fd.readline()
    return first_line


def extract_filename(file_string):
    return file_string.partition("\\")[2]


def extract_file_id(file_name):
    return int(file_name[:file_name.find("_")])  #from start to one before _ [start:end]


def extract_file_rating(file_name):
    return int(file_name[file_name.find("_")+1:file_name.find(".")])


def standardize_ratings(rating):
    if(rating >= 7):
        return 1
    elif(rating <= 4):
        return 0


def pos_tagging(doc):
    my_doc = nlp(doc)
    pos_list = []
    #$lemma = []
    #$non_stopwords = []

    for token in my_doc:
        pos_list.append(token.pos_)
        #$lemma.append(token, lemma_)
        #$if token.is_stop:
            #$continue
        #$else:
            #$non_stopwords.append(token.text)


        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    return pos_list



def lemmatization(doc):
    my_doc = nlp(doc)

    #replace all words by their lemma, if no lemma is found, the original word is returned
    #print("original: ",my_doc)
    tokens = [token.lemma_ for token in my_doc]
    #print(tokens)
    return tokens
    

def sentence_segmentation(doc):
    #nlp = spacy.load('en_core_web_sm')
    doc2 = nlp(doc)
    for sent in doc2.sents:
        print(sent.text)


def stop_word_removal(doc):
    #nlp = English()
    #  "nlp" Object is used to create documents with linguistic annotations.
    my_doc = nlp(doc)

    #create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    from spacy.lang.en.stop_words import STOP_WORDS
    #Create list of word tokens after removing stopwords
    filtered_sentence = []

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)
    #print(token_list)
    print("method 1: ")
    print(filtered_sentence)

def stop_word_removal_2(doc):
    #nlp = spacy.load('en_core_web_sm')

    #spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    #print(spacy_stopwords)

    my_doc = nlp(doc)
    tokens = [token.text for token in my_doc if not token.is_stop]
    #print(tokens)
    return tokens


def clean_reviews(review):
    review = review.replace("<br />", "\n")
    return review


def create_bigram(doc):
    #create a list for the result
    result = list()
    #create a list that contains no punctuation
    sentence = list()

    #parse through the document to add all tokens that are words to the sentence list
    for token in doc:
        if token.is_alpha:
            sentence.append(token)

    #parse through the sentence while adding words in groups of two to the result
    for word in range(len(sentence) -1):
        firstWord = sentence[word]
        secondWord = sentence[word + 1]
        element = [firstWord, secondWord]
        result.append(element)

    return result


def extract_lexical_features_to_dict():
    with open(lexiconAsCsv,'r') as f:
        lexicon = list(csv.reader(f))
        
        #create dictionary out of lexicon with word as key and valence, arousal and dominance as values
        for id, row in enumerate(lexicon):
            #my_dict[key] = "value" https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/
            lexicon_dict[lexicon[id][wordIndex]] = (lexicon[id][valenceIndex], lexicon[id][arousalIndex], lexicon[id][dominanceIndex])
            if(id == 0): #TODO: remove this testline
                print((lexicon[id][valenceIndex], lexicon[id][arousalIndex], lexicon[id][dominanceIndex]))

################## preparation ##################

#load lexicon to lexicon_dict
extract_lexical_features_to_dict()

#load files
txt_files = glob.glob(rel_path_to_trainingdata+"*.txt")
#print(txt_files)

#extract filename, id and rating
for file in txt_files:
    filename = extract_filename(file)
    standardizedRating = standardize_ratings(extract_file_rating(filename))
    cleanedReview = clean_reviews(read_first_line_txt(rel_path_to_trainingdata+filename))
    overall_dict[extract_file_id(filename)] = (standardizedRating,cleanedReview)

################## preprocessing ##################
preproc_list = []

#lemmatization, pos-tagging
for key, value in overall_dict.items():
    review = value[1]
    sentence_list = []
    doc = nlp(review)
    sentences = list(doc.sents)
    for sentence in sentences:
        preproc_list = []
        for token in sentence:
            text = token.text
            lemma = token.lemma_
            pos_tag = token.pos_
            preproc_list.append((text, lemma, pos_tag))
        sentence_list.append((sentence, preproc_list))
    overall_dict[key] = (value[0], value[1], sentence_list)
    
with open(rel_path_to_trainingdata+"preprocessing_pickled.py", "wb") as outfile:
    pickle.dump(overall_dict, outfile)

# print(preproc_list[0])
# print(preproc_list[1])


################## preprocessing with lexicon ##################
preproc_list_lexical = []

#lemmatization, pos-tagging, extraction valence, arousal, dominance, count num pos. words, count num neg. words
# for key, value in overall_dict.items():
#     review = value[1]
#     preproc_list = []
#     lexical_list = [] # [valence, arousal, dominance]
#     positiveWordCount = 0
#     negativeWordCount = 0
#     doc = nlp(review)
#     for token in doc:
#         text = token.text
#         lemma = token.lemma_
#         pos_tag = token.pos_
#         valence = None
#         arousal = None
#         dominance = None
        
#         if text in lexicon_dict:
#             valence = lexicon_dict[text][0]
#             arousal = lexicon_dict[text][1]
#             dominance = lexicon_dict[text][2]
#             if(float(valence) > 5):
#                 positiveWordCount += 1
#             elif(float(valence) < 5):
#                 negativeWordCount += 1
#         preproc_list.append((text, lemma, pos_tag))
#         lexical_list.append((valence, arousal, dominance))
        
#     overall_dict[key] = (value[0], value[1], preproc_list, lexical_list, positiveWordCount, negativeWordCount)

# with open(rel_path_to_trainingdata+"preprocessing_pickled.py", "wb") as outfile:
#     pickle.dump(overall_dict, outfile)
    #print(lexical_list)





#MaLe Sentiment Analysis:
#unigram not in order of occurence, with counting
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(review_list)
# feature_names = vectorizer.get_feature_names()
# feature_vector = X.toarray()
# #print(feature_names)
# #print(feature_vector)


# #bigram not in order of occurence, with counting
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2))
# X2 = vectorizer2.fit_transform(review_list_cleaned)
# feature_names2 = vectorizer2.get_feature_names()
# feature_vector2 = X2.toarray()
#print(feature_names2)
#print(feature_vector2)