import numpy as np
import pandas as pd
import glob
import spacy
import pickle
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile, f_classif
import json
import time

start_time = time.time()
#rel_path_to_trainingdata = 'D:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/shortStories/'#'partOfData/'
# rel_path_to_trainingdata = 'partOfData/'
rel_path_to_trainingdata = 'D:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/IMDb-Datenset_Training_Testing/10000_test/'

rel_path_to_additionaldata = 'additionalData/'
nlp = spacy.load('en_core_web_sm')

######## variables for lexical resource ########
lexicon = []
lexiconAsCsv = 'WarrinerList2013.csv' # name of lexicon to use
wordIndex = 1 # index of wordtext in csv
valenceIndex = 2 # index of valence in csv
arousalIndex = 5 # index of arousal in csv
dominanceIndex = 8 # index of dominance in csv
lexicon_dict = {} # {'Word': ('V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum'), 'aardvark': ('6.26', '2.41', '4.27'), ... }
negationWordList = ["hardly", "cannot", "lack", "lacking", "lacks", "neither", "nor", "without", "not", "n\'t"]
overallFrequencyPositive = {}
overallFrequencyPositive['texts'] = []
overallFrequencyPositiveAndNegative = {} # will be filled in extract_lexiconbased_features
overallFrequencyPositiveAndNegative['texts'] = []
overallNumSentencesPosNegNeut = {}
overallNumSentencesPosNegNeut['texts'] = []
wordsPerDocumentCount = []

################## preparation steps ##################

# load lexicon from pickle
with open(rel_path_to_additionaldata+"lexicon_pickled.py" , "rb") as infile:
    lexicon_dict = pickle.load(infile)

# load cleaned reviewdata from pickle
with open(rel_path_to_additionaldata+"overall_dict_pickled.py" , "rb") as infile:
    overall_dict = pickle.load(infile)

# create pandas dataframe
df = pd.DataFrame(overall_dict)
# append labels to dataframe
labels = df['standardizedRating'].values
# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:], labels, test_size=0.2, random_state=0)

################## extention dataframe ##################
# Method to calculate unigram-vocabulary on X_train (whole data=all train-reviews)
# returns vocabulary as list
def create_unigram_vocabulary_of_trainset(df):
    vocabulary = []
    for index, row in df.iterrows():
        review = row['cleanedReview']
        doc = nlp(review)
        for token in doc:
            if token.is_alpha and not token.is_stop: # only take non-stop words
                word = token.lemma_.lower() # use lemma of word in lowercase
                if word not in vocabulary:
                    vocabulary.append(word)
    return vocabulary


def create_bigram_vocabulary_of_trainset(df):
    word_list = create_unigram_vocabulary_of_trainset(df)
    bigram_vocabulary = []
            
    for word in range(len(word_list) -1):
        firstWord = word_list[word]
        secondWord = word_list[word + 1]
        element = [firstWord, secondWord]
        bigram_vocabulary.append(element)

    return bigram_vocabulary

# Method which calculates the ngram-vector
# returns ngram-vector per review
def calculate_ngram_frequencyvector(vocabulary, ngram):
    ngram_vector = [0] * len(vocabulary)
    
    for token in ngram:
        if token in vocabulary:
            word_index = vocabulary.index(token)
            ngram_vector[word_index] += 1
    return ngram_vector


# Method which creates a vocabulary of unigrams
# returns list of unigrams and all unigrams as one string
def create_unigram_list(review):
    #create a list for the bigram_list
    unigram_list = []
    doc = nlp(review)
    for token in doc:
        if token.is_alpha and not token.is_stop:
            unigram_list.append(token.lemma_.lower())

    unigram_string = " "
    unigram_string = unigram_string.join(unigram_list)
    return (unigram_list, unigram_string)

# Method which creates a vocabulary of bigrams
# returns list of bigrams
def create_bigram_list(review):
    #create a list for the bigram_list
    bigram_list = []
    #create a list that contains no punctuation
    (word_list, unigram_string) = create_unigram_list(review)

    #parse through the sentence while adding words in groups of two to the bigram_list
    for word in range(len(word_list) -1):
        firstWord = word_list[word]
        secondWord = word_list[word + 1]
        element = [firstWord, secondWord]
        bigram_list.append(element)

    return (bigram_list)

# Lemmatization, pos-tagging
def pos_tagging(text):
    text_tokens = []
    lemma_string = ""
    lemma_tokens = []
    pos_string = ""
    pos_tokens = []
    doc = nlp(text)

    for token in doc:
        text_tokens.append(token.text)
        lemma_tokens.append(token.lemma_)
        lemma_string += token.lemma_+" "
        pos_tokens.append(token.pos_)
        pos_string += token.pos_+" "
    
    return (text_tokens, lemma_string, lemma_tokens, pos_string, pos_tokens)
    
# Extract features pos counts
def count_posTags(pos_list):
    noun_count = 0
    verb_count = 0
    adv_count = 0
    adj_count = 0
    punct_count = 0
    for token in pos_list:
        if token == "NOUN" or token == "PROPN":
            noun_count += 1
        elif token == "VERB":
            verb_count += 1
        elif token == "ADV":
            adv_count += 1
        elif token == "ADJ":
            adj_count += 1
        elif token == "PUNCT":
            punct_count += 1
    
    return (noun_count, verb_count, adv_count, adj_count, punct_count)

# Extract features elongated words
def count_elongated_words(review):
    elongated_word_count = 0
    doc = nlp(review)
    for token in doc:
        word = token.text.lower()
        sameCharacterCount = 1
        currentCharacter = word[0]
        if len(word) > 2:
            for i in range (len(word)):
                if i < (len(word)-1): # if i is max second to last character of string
                    currentCharacter = word[i]
                    if currentCharacter == word[i+1]:
                        sameCharacterCount += 1
                    else:
                        sameCharacterCount = 1 # rebase to account for more than one double character/word
                    if sameCharacterCount > 2:
                        elongated_word_count += 1
                        break
    return elongated_word_count


# Create list containing the indexes of all words in a review,
# for which valence would have to be flipped to the opposite, if available
def findNegationContexts(sentence):
    tempNegCtx_list = []
    negationCtx_list = []
    exceptionList1 = ["not only", "not just", "no question", "not to mention", "no wonder"]
    for i in range(len(sentence)):
        bigram = ''
        trigram = ''
        if sentence[i].lemma_.lower() in negationWordList:
            if i < len(sentence):
                j = i + 1
                while j < (len(sentence)-1) and sentence[j].pos_ != "PUNCT":
                    negationCtx_list.append(j) # append index of word between negWord and next punctuation
                    j += 1
                # check first exception
                if i < (len(sentence)-1):
                    bigram = f'{sentence[i].lemma_} {sentence[i+1].lemma_}'
                if i < (len(sentence)-2):
                    trigram = f'{sentence[i].lemma_} {sentence[i+1].lemma_} {sentence[i+2].lemma_}'
                if bigram != '' and bigram.lower() in exceptionList1:
                    tempNegCtx_list = [] # negation context does not apply
                    continue
                if trigram != '' and trigram.lower() in exceptionList1:
                    tempNegCtx_list = [] # negation context does not apply
                    continue

                if j < len(sentence):
                    # check second exception
                    if sentence[j].text == "?" and checkNegationException(sentence, i) == True:
                        tempNegCtx_list = [] # negation context does not apply
                    else:
                        negationCtx_list.extend(tempNegCtx_list) # true negation context, extend negationCtx_list with previously found indexes
                        tempNegCtx_list = []
    return negationCtx_list

def checkNegationException(sentence, i):
    # if word in the first three places of a sentence at the beginning of a text
    if i in (0, 1, 2):
        return True
    # if word in the first three places of a sentence in the middle of a text
    elif sentence[i-1].pos_ == "PUNCT":
        return True
    elif sentence[i-2].pos_ == "PUNCT":
        return True
    elif sentence[i-3].pos_ == "PUNCT":
        return True
    else:
        return False

# Extract lexiconbased features
def extract_lexiconbased_features(review):
    lexical_list = [] # [valence, arousal, dominance]
    numNegationContexts = 0
    totalDocumentWordCount = 0
    totalPositiveWordCount = 0
    totalNegativeWordCount = 0
    totalWordCount = 0
    positiveSentences = 0
    negativeSentences = 0
    neutralSentences = 0
    valence_values = []
    arousal_values = []
    dominance_values = []
    doc = nlp(review)
    sentences = list(doc.sents) # create list of sentences of doc

    for sentence in sentences:
        sentenceWordCount = 0
        positiveWordCount = 0
        negativeWordCount = 0
        negationIndexWordList = findNegationContexts(sentence) # create list of indexes of all words, for which valence has to be flipped
        for i in range(len(sentence)):
            token = sentence[i]
            lemma = token.lemma_
            valence = None
            arousal = None
            dominance = None
            
            if token.is_alpha:
                sentenceWordCount += 1
            if lemma in lexicon_dict:
                valence = lexicon_dict[lemma][0]
                valence = float(valence)
                arousal = lexicon_dict[lemma][1]
                arousal = float(arousal)
                dominance = lexicon_dict[lemma][2]
                dominance = float(dominance)
                # calculation to flip valence around the middle of 5, if the word is affected by negation:
                if i in negationIndexWordList:
                    if valence > 5:
                        valence = 5 - (valence - 5) # flip valence to negative
                    elif valence < 5:
                        valence = 5 + (5 - valence) # flip valence to positive
                    numNegationContexts += 1

                if valence > 5:
                    positiveWordCount += 1
                elif valence < 5:
                    negativeWordCount += 1
            if(valence != None): # only append the values that could be found in lexicon
                valence_values.append(valence)
                arousal_values.append(arousal)
                dominance_values.append(dominance)
        # calculated per sentence
        totalDocumentWordCount += sentenceWordCount        
        totalPositiveWordCount += positiveWordCount
        totalNegativeWordCount += negativeWordCount
        if positiveWordCount > negativeWordCount:
            positiveSentences += 1
        elif negativeWordCount > positiveWordCount:
            negativeSentences += 1
        else:
            neutralSentences += 1
    wordsPerDocumentCount.append(totalDocumentWordCount)
    valence_avg = sum(valence_values)/len(valence_values)
    arousal_avg = sum(arousal_values)/len(arousal_values)
    dominance_avg = sum(dominance_values)/len(dominance_values)
    positiveSentenceFrequency = (100/len(sentences))*positiveSentences
    negativeSentenceFrequency = (100/len(sentences))*negativeSentences
    # percent of positive sentences per text
    overallFrequencyPositive['texts'].append({
        'text': review,
        'positiveSentenceFrequency': positiveSentenceFrequency
    })
    # percent of positive and negative sentences per text
    overallFrequencyPositiveAndNegative['texts'].append({
        'text': review,
        'positiveSentenceFrequency': positiveSentenceFrequency,
        'negativeSentenceFrequency': negativeSentenceFrequency
    })
    # optional result file
    overallNumSentencesPosNegNeut['texts'].append({
        'text': review,
        'numPosSentences': positiveSentences,
        'numNegSentences': negativeSentences,
        'numNeutSentences': neutralSentences
    })
    return (valence_avg, arousal_avg, dominance_avg, numNegationContexts, totalPositiveWordCount, totalNegativeWordCount)

# prepare feature lists and append to df
def extendDataframe(df):
    print('Extending Dataframe, preparing features...')
    unigram_list = []
    unigram_string_list = []
    bigram_list = []
    text_tokens = []
    lemma_strings = []
    lemma_tokens = []
    pos_strings = []
    pos_tokens = []
    noun_count = []
    verb_count = []
    adverb_count = []
    adjective_count = []
    punctuation_count = []
    elongatedWord_counts = []
    valence_avg = []
    arousal_avg = []
    dominance_avg = []
    negation_contexts_count = []
    positive_word_count = []
    negative_word_count = []
    for index, row in df.iterrows():
        print(index)
        (unigram, unigram_string) = create_unigram_list(row['cleanedReview']) # (unigram, unigram_string, unigram_frequency)
        (bigram) = create_bigram_list(row['cleanedReview']) # (bigram, bigram_string, bigram_frequency)
        unigram_list.append(unigram)
        unigram_string_list.append(unigram_string)
        bigram_list.append(bigram)
        (tt, ls, lt, ps, pt) = pos_tagging(row['cleanedReview'])
        (nc, vc, advc, adjc, punctc) = count_posTags(pt) # count certain pos tags
        ewc = count_elongated_words(row['cleanedReview'])
        (valence, arousal, dominance, numNegationContexts, wordCountPos, wordCountNeg) = extract_lexiconbased_features(row['cleanedReview']) # use text tokens
        text_tokens.append(tt)
        lemma_strings.append(ls)
        lemma_tokens.append(lt)
        pos_strings.append(ps)
        pos_tokens.append(pt)
        noun_count.append(nc)
        verb_count.append(vc)
        adverb_count.append(advc)
        adjective_count.append(adjc)
        punctuation_count.append(punctc)
        elongatedWord_counts.append(ewc)
        valence_avg.append(valence)
        arousal_avg.append(arousal)
        dominance_avg.append(dominance)
        negation_contexts_count.append(numNegationContexts)
        positive_word_count.append(wordCountPos)
        negative_word_count.append(wordCountNeg)

    # extend dataframe with feature lists
    df['unigramFrequency'] = unigram_string_list
    df['bigramFrequency'] = unigram_string_list # TfidfVectorizer needs unigram list
    df['cleanedReviewTokenized'] = text_tokens
    df['lemmaStrings'] = lemma_strings
    df['lemmaTokens'] = lemma_tokens
    df['posStrings'] = pos_strings
    df['posTokens'] = pos_tokens
    df['nounCounts'] = noun_count
    df['verbCounts'] = verb_count
    df['adverbCounts'] = adverb_count
    df['adjectiveCounts'] = adjective_count
    df['punctuationCounts'] = punctuation_count
    df['elongatedWord_counts'] = elongatedWord_counts
    df['valence'] = valence_avg
    df['arousal'] = arousal_avg
    df['dominance'] = dominance_avg
    df['negationContextsCount'] = negation_contexts_count
    df['positiveWordCount'] = positive_word_count
    df['negativeWordCount'] = negative_word_count
    print(f'df Grösse: {df.size}')
    return df

################## Settings - Training ##################
print(f'X_train vor extendDataframe: {X_train.size}')
X_train = extendDataframe(X_train)
print(f'X_train nach extendDataframe: {X_train.size}')

################## Settings - Testing ################## 
print(f'X_test vor extendDataframe: {X_test.size}')
X_test = extendDataframe(X_test)
print(f'X_test nach extendDataframe: {X_test.size}')

################## pipeline ##################
# comment out all features, that should not be considered for classification
mapper = DataFrameMapper ([
    #  ('unigramFrequency', TfidfVectorizer(min_df=5)),
    #  ('bigramFrequency', TfidfVectorizer(ngram_range=(2,2), min_df=5)),
    #  ('elongatedWord_counts', None),
    #  ('nounCounts', None), # doesnt need transformation
    #  ('verbCounts', None),
    #  ('adverbCounts', None),
    #  ('adjectiveCounts', None),
    #  ('punctuationCounts', None),
    #  ('negationContextsCount', None),
    #  ('positiveWordCount', None),
    #  ('negativeWordCount', None),
     ('valence', None),
    #  ('arousal', None),
    #  ('dominance', None)
])

# initialize Training
print('Transforming trainingdata...')
transformed_train = mapper.fit_transform(X_train)
print('Elapsed Time: %s seconds' % (time.time() - start_time))
# Initialize Testing
print('Transforming testdata...')
transformed_test = mapper.transform(X_test)
print('Elapsed Time: %s seconds' % (time.time() - start_time))

classifier = SVC()
print('Fit data...')
classifier.fit(transformed_train, y_train)
print('Predicting sentiment values...')
predictions = classifier.predict(transformed_test)
print(classification_report(y_test, predictions))

time_minutes = (time.time() - start_time)/60
print('Execution time in seconds: %s seconds' % (time.time() - start_time))
print('Execution time in minutes: %s minutes' % time_minutes)

#pickle model and vectorizer
with open('model_file.py', 'wb') as model_pickle, open('vectorizer.py', 'wb') as vectorizer_pickle:
        pickle.dump(classifier, model_pickle)
        pickle.dump(mapper, vectorizer_pickle)

################## create result-files ##################
# calculate avg length of words per document
avgWordsPerDocument = 0
totalLengthWords= 0
for i in wordsPerDocumentCount:
    totalLengthWords += i
avgWordsPerDocument = (totalLengthWords/len(wordsPerDocumentCount))
print(f'Avg words per document: {avgWordsPerDocument}')

# FA-5 Creates a list with all reviews and per review the number of positive, negative and neutral sentences
print('Write result-file \'numSentencesPosNegNeut.txt\'...')
with open('numSentencesPosNegNeut.txt', 'w') as outfile:
    json.dump(overallNumSentencesPosNegNeut, outfile, indent=4)

# FA-6 creates a list with all reviews and per review the number of positive sentences in percent
print('Write result-file \'reviewFrequencyPositive.txt\'...')
with open('reviewFrequencyPositive.txt', 'w') as outfile:
    json.dump(overallFrequencyPositive, outfile, indent=4)

# FA-7 creates a list with all reviews and per review the number of positive and negative sentences in percent
print('Write result-file \'reviewFrequencyPositiveAndNegative.txt\'...')
with open('reviewFrequencyPositiveAndNegative.txt', 'w') as outfile:
    json.dump(overallFrequencyPositiveAndNegative, outfile, indent=4)