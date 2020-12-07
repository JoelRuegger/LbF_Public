#!/usr/bin/env python3
'''classifies a data set given a classifier model'''

import argparse
import pandas as pd
import pickle
import train_classifier as tc



def append_feature_columns(train_data_transformed: pd.DataFrame, function, columname: str, function_argument: object) -> tuple:
    '''
    Append new columns with features to the pandas dataframe
    
    arguments:
        train_data_transformed: Train data containing all created features
        function: The function to be applied to values of the column
        columname: Name of the column to be appended
        function_argument: Pass arguments to the functions map_calgari etc.
        
    returns: Updated versions of train_data_transformed and test_data_transformed with the new feature column.
    '''

    train_map = train_data_transformed.copy()
    if function == tc.map_calgari:
        train_map['Text'] = train_map['Text'].apply(function, calgari_list=function_argument)
    elif function == tc.apply_bigram_frequency:
        train_map['Text'] = train_map['Text'].apply(function, bigram_list=function_argument)
    elif function == tc.neg_polex_matches:
        train_map['Text'] = train_map['Text'].apply(function, polex=function_argument)
    # in order to apply functions with no arguments
    else:
        train_map['Text'] = train_map['Text'].apply(function)

    train_map = train_map.rename(columns={'Text': columname})
    train_data_transformed = train_data_transformed.join(train_map[columname])

    return train_data_transformed
    
def classify(input, output, model, vectorizer, calgari=None):
    data = tc.read_csv(input)
    
    with open("./data/polex_de_clean_pickle.py", 'rb') as polex_pickle:
        polex = pickle.load(polex_pickle)
        polex = dict(zip(polex.lemma, polex.sentiment))
    
    if calgari != None:
        with open(calgari, 'rb') as calgari_pickle:
            global calgari_tokens 
            calgari_tokens = pickle.load(calgari_pickle)
    
        data = append_feature_columns(data, tc.map_calgari, 'calgarimatches', calgari_tokens)
    data = append_feature_columns(data, tc.extract_adj_list, 'adjmatches', None)
    data = append_feature_columns(data, tc.average_word_length, 'avgwordlength', None)
    data = append_feature_columns(data, tc.average_word_length, 'neg_polex_matches', None)
    
    
    with open(model, 'rb') as model_pickle, open(vectorizer, 'rb') as vec_pickle:
        vec = pickle.load(vec_pickle)
        fitted_data = vec.transform(data)
        classifier = pickle.load(model_pickle)
        predictions = classifier.predict(fitted_data)
        print("\nData set classified!")
        tc.write_scores(output,predictions)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help='input data to be classified', type=str, required=True)
    argparser.add_argument('-o', '--output', help='classified output', type=str, required=True)
    argparser.add_argument('-m', '--model', help='pickled classifier model', type=str, required=True)
    argparser.add_argument('-v', '--vectorizer', help='pickled vectorizer', type=str, required=True)
    argparser.add_argument('-c', '--calgari', help='pickled calgari tokens', type=str, required=False)
    
    args = argparser.parse_args()
    
    data = tc.read_csv(args.input)
    
    with open("./classification/data/polex_de_clean_pickle.py", 'rb') as polex_pickle:
        polex = pickle.load(polex_pickle)
        polex = dict(zip(polex.lemma, polex.sentiment))
    
    if args.calgari != None:
        with open(args.calgari, 'rb') as calgari_pickle:
            global calgari_tokens 
            calgari_tokens = pickle.load(calgari_pickle)
    
        data = append_feature_columns(data, tc.map_calgari, 'calgarimatches', calgari_tokens)
    data = append_feature_columns(data, tc.extract_adj_list, 'adjmatches', None)
    data = append_feature_columns(data, tc.average_word_length, 'avgwordlength', None)
    data = append_feature_columns(data, tc.average_word_length, 'neg_polex_matches', None)
    
    
    with open(args.model, 'rb') as model_pickle, open(args.vectorizer, 'rb') as vec_pickle:
        vec = pickle.load(vec_pickle)
        fitted_data = vec.transform(data)
        classifier = pickle.load(model_pickle)
        predictions = classifier.predict(fitted_data)
        print("\nData set classified!")
        tc.write_scores(args.output,predictions)
    
    
    
if __name__ == '__main__':
    main()