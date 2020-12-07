#!/usr/bin/env python3

""" train a classification baseline"""

import argparse
import csv
import numpy as np
import pandas as pd
import pickle
import spacy
import re

from sklearn_pandas import DataFrameMapper

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC



def read_csv(filename):
    data = pd.read_csv(filename, encoding='utf-8', quotechar="'")
    return data
    
    
def write_scores(filename, predictions):
    predictions = [(i+1,j) for i,j in enumerate(predictions)]
    with open(filename,'w') as resultof:
        csv_writer = csv.writer(resultof,delimiter=",")
        csv_writer.writerow(['Id','Prediction'])
        for id_,pred in predictions:
            csv_writer.writerow([id_,pred])
            
            
class DataFrameColumnExtracter(TransformerMixin):
    '''selects a specific column in the pandas dataframe to then be fed into a transformer'''
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]

    # To support grid search:
    def get_params(self):
        return None


###########################################################
def average_word_length(text):
    '''' Calculate the average word length in a given text '''
    sum = 0.0
    count = 0
    for word in text.split(" "):
        sum += len(word)
        count += 1
    return (sum / count)


###########################################################

def extract_adj_list(text: str):
    '''
    extracts all tokens that are an adjective
    
    arguments:
        text:   plain text
        
    returns: list of all adjectives
    '''
    nlp = spacy.load("de_core_news_sm")
    spacy_text = nlp(text)
    adjectives = [token.text for token in spacy_text if token.pos_ == "ADJ"]
    return (" ").join(adjectives)


############################################################
def list_of_bigrams(data_in: pd.DataFrame):
    '''
    Generate a list of all bigrams with their frequencies.
    
    arguments:
        data_in: Dataframe containing the text and its label
    
    returns: Dictionary containing each category as key, and a list of bigrams, sorted by their frequency in that category
    '''

    # contains tuples of the form (category, sentence)
    category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]

    # build dict {category: [tokens]}
    category_tokens = {0.0: {}, 1.0: {}}
    nlp = spacy.load("de_core_news_sm")
    for elem in category_text:
        # count frequencies for all occuring bigrams
        spacy_text = nlp(elem[1])
        tokenized_text = [token.text for token in spacy_text]
        for bigram in zip(*[tokenized_text[i:] for i in range(2)]):
        #for bigram in zip(*[list(elem[1][i:]) for i in range(2)]):
            dict = category_tokens[elem[0]]
            key = bigram[0] + bigram[1]
            if key in dict:
                dict[key] = dict[key] + 1
            else:
                dict[key] = 1

    Labels = [0.0, 1.0]
    category_lists = {0.0: [], 1.0: []}
    for label in Labels:
        category_lists[label] = sorted(category_tokens[label].keys(), key=lambda k: category_tokens[label][k])
    return category_lists


def get_bigram_frequency_list(text: str) -> str:
    '''
    arguments:
        text: The text to be searched through.
    
    returns: List of bigrams sorted by how frequent they are in the sentence.
    '''
    bigram_dict = {}
    nlp = spacy.load("de_core_news_sm")
    spacy_text = nlp(text)
    tokenized_text = [token.text for token in spacy_text]
    #for bigram in nltk.bigrams(tokenized_text):
    for bigram in zip(*[tokenized_text[i:] for i in range(2)]):
        key = bigram[0] + bigram[1]
        if key in bigram_dict:
            bigram_dict[key] = bigram_dict[key] + 1
        else:
            bigram_dict[key] = 1
    bigram_list = sorted(bigram_dict.keys(), key=lambda k: bigram_dict[k])
    return bigram_list


def apply_bigram_frequency(text: str, bigram_list: list) -> float:
    '''
    Calculate error for bigrams in a text compared to a lists of bigrams inspired by Canvar & Trenkle 1994, N-Gram-Based Text Categorization
    
    arguments:
    sentence: The text to be searched.
    bigram_list: List of bigrams ordered by some frequency.
    
    returns: Represents the error.
    '''
    sentence_bigrams = get_bigram_frequency_list(text)
    if len(sentence_bigrams) == 0:
        return "0"
    out_of_place = 0
    max_value = len(bigram_list)
    max_index = len(bigram_list)

    for s_index, sentence_bigram in enumerate(sentence_bigrams):
        found = False
        for bigram_index, bigram in enumerate(bigram_list[0: max_index]):
            if bigram != sentence_bigram:
                continue
            else:
                out_of_place += abs(s_index - bigram_index)
                found = True
                break
        if not found:
            out_of_place += max_value

    result = out_of_place / len(text)
    return result
#########################################################

def neg_polex_matches(text: str, polex):
    '''
    calculates per text how many tokens with negative polarity are present.
    
    arguments:
        text:       plaintext where the adjectives shall be matched
        polex:      a dictionary of the polar lexicon of the form {lemma: sentiment}
        
    returns: number of adjectives with negative polarity
    '''
    nlp = spacy.load("de_core_news_sm")
    spacy_text = nlp(text)
    counter = 0
    for token in spacy_text:
        if token.text in polex.keys():
            if polex[token.text] == 'NEG':
                counter += 1
    return counter

###################### CALGARI ##########################

def get_term_freq_per_cat(dict, cat, token):
    if (cat, token) in dict.keys():
        return dict[(cat, token)]
    else:
        return 0

def calgari(data_in: pd.DataFrame) -> list:
    #inspired by http://brenthecht.com/papers/bhecht_chi2011_location.pdf
    '''
    calculates the 200 best calgari tokens based on the training data
    arguments:
        data_in: pandas dataframe containing training data
    
    returns: The 200 best calgari tokens
    '''

    # contains tuples of the form (category, sentence)
    category_text = [(c, s) for c, s in zip(data_in['Label'].values, data_in['Text'].values)]
    # tokenize sentences
    category_tokens = []
    nlp = spacy.load("de_core_news_sm")
    for elem in category_text:
        sentence = nlp(elem[1])
        tokens = [token.text for token in sentence]
        category_tokens.append((elem[0], tokens))

    # structure: {category: freq}
    term_freq = {}
    # structure: {(category, token):freq}
    term_freq_per_category = {}
    term_count = 0
    term_count_per_category = {1.0: 0, 0.0: 0}

    for cat, text in category_tokens:
        for token in text:
            if token in term_freq.keys():
                term_freq[token] += 1
            else:
                term_freq[token] = 1
            if (cat, token) in term_freq_per_category.keys():
                term_freq_per_category[(cat, token)] += 1
            else:
                term_freq_per_category[(cat, token)] = 1

            term_count += 1
            term_count_per_category[cat] += 1

    # structure: [(calgari value, tok)]
    output = []

    for tok, freq in term_freq.items():
        if freq > 2:
            # max(probability t given category: termfrequency in category/total amount of terms in category)
            upper_fraction = max(
                (get_term_freq_per_cat(term_freq_per_category, 1.0, tok) / term_count_per_category[1.0]),
                (get_term_freq_per_cat(term_freq_per_category, 1.0, tok) / term_count_per_category[0.0]))
            # probability term: termfrequency/total amount of terms
            lower_fraction = freq / term_count
            output.append((upper_fraction / lower_fraction, tok))

    sorted_output = sorted(output, reverse=True)
    # returns 200 best calgari tokens
    return [re.sub("\(", "\(", re.sub("\)", "\)", tok)) for val, tok in sorted_output[:200]]


def map_calgari(text: str, calgari_list: list) -> str:
    '''
    matches the calgary token list to an actual text
    
    arguments:
        text: text to be searched.
        calgari_list: List of calgari-tokens that should be matched
    
    returns: all the substrings that match elements from the calgari_list
    '''
    output = []
    for tok in calgari_list:
        if tok == ".":
            continue
        else:
            matches = re.findall(tok, text)
            if matches == []:
                continue
            else:
                output.extend(matches)
    return (" ").join(output)
#################################################################


def append_feature_columns(train_data_transformed: pd.DataFrame, test_data_transformed: pd.DataFrame, function, columname: str, function_argument: object) -> tuple:
    '''
    Append new columns with features to the pandas dataframe
    
    arguments:
        train_data_transformed: Train data containing all created features
        test_data_transformed: Test data containing all created features
        function: The function to be applied to values of the column
        columname: Name of the column to be appended
        function_argument: Pass arguments to the functions map_calgari etc.
        
    returns: Updated versions of train_data_transformed and test_data_transformed with the new feature column.
    '''

    train_map = train_data_transformed.copy()
    if function == map_calgari:
        train_map['Text'] = train_map['Text'].apply(function, calgari_list=function_argument)
    elif function == apply_bigram_frequency:
        train_map['Text'] = train_map['Text'].apply(function, bigram_list=function_argument)
    elif function == neg_polex_matches:
        train_map['Text'] = train_map['Text'].apply(function, polex=function_argument)
    # in order to apply functions with no arguments
    else:
        train_map['Text'] = train_map['Text'].apply(function)

    train_map = train_map.rename(columns={'Text': columname})
    train_data_transformed = train_data_transformed.join(train_map[columname])

    test_map = test_data_transformed.copy()
    if function == map_calgari:
        test_map['Text'] = test_map['Text'].apply(function, calgari_list=function_argument)
    elif function == apply_bigram_frequency:
        test_map['Text'] = test_map['Text'].apply(function, bigram_list=function_argument)
    elif function == neg_polex_matches:
        test_map['Text'] = test_map['Text'].apply(function, polex=function_argument)
    else:
        test_map['Text'] = test_map['Text'].apply(function)

    test_map = test_map.rename(columns={'Text': columname})
    test_data_transformed = test_data_transformed.join(test_map[columname])
    
    return train_data_transformed, test_data_transformed


def classify(train_data, test_data, model_file:str, vectorizer: str, grid_search=False):
    '''
    trains a classifier and returns the predictions
    
    arguments:
        train_data:     a pandas dataframe containing the training data
        test_data:      a pandas dataframe containing the test data
        model_file:    file where the model shall be pickled to 
        vectorizer:     file where the vecotrizer shall be pickled to
        grid_search:    a boolean flag to induce grid search
        
    returns: the predictions
    '''
    train_labels = train_data['Label'].values
    test_labels = test_data['Label'].values
    

    
    #handles the different columns and applies the transformations if needed
    mapper = DataFrameMapper ([
            ('Text', TfidfVectorizer()),
            ('calgarimatches', TfidfVectorizer()),
            ('adjmatches', TfidfVectorizer()),
            ('avgwordlength', None), # doesn't need a  transformation
            ('neg_polex_matches', None),
            #('bigram_frequency_0.0', None),
            #('bigram_frequency_1.0', None)
    ])
    
    transformed_train = mapper.fit_transform(train_data)
    transformed_test = mapper.transform(test_data)
    
    classifier = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=5, tol=None, class_weight={0.0:0.2, 1.0:1})
    
    #classifier = LinearSVC(class_weight='balanced')
    #classifier = SVC(class_weight={0.0:0.2, 1.0:1})
    
    
    if grid_search == True:
        parameters_sgd = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss'],'max_iter':[3,6], 'class_weight': [{0.0: n} for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}
        parameters_linearsvc = {'penalty': ['l1', 'l2'], 'max_iter': [3,10], 'class_weight': [{0.0: n} for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}
        parameters_svc = {'kernel': ['linear', 'poly', 'rbf'], 'class_weight': [{0.0: n} for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}

        gs = GridSearchCV(classifier, parameters_sgd)
        #gs = GridSearchCV(classifier, parameters_linearsvc)
        #gs = GridSearchCV(classifier, parameters_svc)
        gs.fit(transformed_train, train_labels)
        #best_model = gs.fit(transformed_train, train_labels)
        print("Best parameters set found on development set:")
        #print(best_model.best_estimator_.get_params()['clf'])
        print(gs.best_params_)
    
    classifier.fit(transformed_train, train_labels)
    predictions = classifier.predict(transformed_test)
    
    print(classification_report(test_labels, predictions))
    
    with open(model_file, 'wb') as model_pickle, open(vectorizer, 'wb') as vec_pickle:
        pickle.dump(classifier, model_pickle)
        pickle.dump(mapper, vec_pickle)
    
    return predictions


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', help='training file', type=str, required=True)
    argparser.add_argument('--test', help='test file', type=str, required=True)
    argparser.add_argument('--result', help='result file', type=str, required=True)
    argparser.add_argument('--model', help='pickle to which the classifier model shall be saved', type=str, required=True)
    argparser.add_argument('--vectorizer', help='pickle to which the vectorizer shall be saved', type=str, required=True)
    argparser.add_argument('--calgari', help='file to save the best calgari tokens to', type=str, required=True)
    
    args = argparser.parse_args()
    
    train_data_transformed = read_csv(args.train)
    test_data_transformed = read_csv(args.test)
    
    #compute the 200 best calgari tokens
    calgari_tokens = calgari(train_data_transformed)
    with open(args.calgari, 'wb') as pickle_file:
        pickle.dump(calgari_tokens, pickle_file)
    
    bigrams = list_of_bigrams(train_data_transformed)
    
    with open("./data/polex_de_clean_pickle.py", 'rb') as polex_pickle:
        polex = pickle.load(polex_pickle)
        polex = dict(zip(polex.lemma, polex.sentiment))
    
    
    
    train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed,  map_calgari, 'calgarimatches', calgari_tokens)
    train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed,  extract_adj_list, 'adjmatches',  None)
    train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, average_word_length, 'avgwordlength', None)
    train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, neg_polex_matches, 'neg_polex_matches', polex)
    #train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_0.0', function_argument=bigrams[0.0])
    #train_data_transformed, test_data_transformed = append_feature_columns(train_data_transformed, test_data_transformed, apply_bigram_frequency, 'bigram_frequency_1.0', function_argument=bigrams[1.0])
    
    #print(train_data_transformed)
    #predictions = classify(train_data_transformed, test_data_transformed, args.model, args.vectorizer, True)
    predictions = classify(train_data_transformed, test_data_transformed, args.model, args.vectorizer)
    write_scores(args.result,predictions)
    
    
    
if __name__ == '__main__':
    main()