import numpy as np
import pandas as pd
import glob
import spacy
from spacy.lang.en import English
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import csv
import pickle
import time

rel_path_to_trainingdata = 'D:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/IMDb-Datenset_Training_Testing/10000_test/'
# rel_path_to_trainingdata = 'partOfData/'
rel_path_to_testdata = 'shortStories/'

start_time = time.time()

rel_path_to_additionaldata = 'additionalData/'
overall_dict = {'id': [],
                'standardizedRating' : [],
                'cleanedReview' : []}
external_testset_dict = {'id': [],
                'standardizedRating' : [],
                'cleanedReview' : []}

nlp = spacy.load('en_core_web_sm')


######## variables for lexicon ########
lexicon = []
lexiconAsCsv = 'WarrinerList2013.csv' # name of lexicon to use
wordIndex = 1 # index of wordtext in csv
valenceIndex = 2 # index of valence in csv
arousalIndex = 5 # index of arousal in csv
dominanceIndex = 8 # index of dominance in csv
lexicon_dict = {} # {'Word': ('V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum'), 'aardvark': ('6.26', '2.41', '4.27'), ... }

######## variables for external testset ########
testsetAsCsv = 'EmotionalShortStories_COST2017_CH-DE_list.csv' # name of testset to use

################## preparation lexicon ##################

def extract_lexical_features_to_dict():
    with open(lexiconAsCsv,'r') as f:
        lexicon = list(csv.reader(f))
        
        #create dictionary out of lexicon with word as key and valence, arousal and dominance as values
        for id, row in enumerate(lexicon):
            if id == 0: #skip title
                continue
            lexicon_dict[lexicon[id][wordIndex]] = (lexicon[id][valenceIndex], lexicon[id][arousalIndex], lexicon[id][dominanceIndex])

################## methods for extention of dataframe ##################

def read_first_line_txt(file):
    with open(file, 'r', encoding="utf8") as fd:
        first_line = fd.readline()
    return first_line

def extract_filename(file_string):
    return file_string.partition("\\")[2]

def extract_file_id(file_name):
    return str(file_name[:file_name.find(".")])  #from start to one before . [start:end] id = whole filename without filetype

def extract_file_rating(file_name):
    return int(file_name[file_name.find("_")+1:file_name.find(".")])

def standardize_ratings(rating):
    if(rating >= 7):
        return 1
    elif(rating <= 4):
        return 0

def clean_reviews(review):
    review = review.replace("<br />", "")
    return review

################## preparation external testset ##################
# load short stories files
test_txt_files = glob.glob(rel_path_to_testdata+"*.txt")

def extract_test_id(file_name):
    return file_name[:file_name.find("-")] # from start to first occurence of -


def extract_test_rating(file_name):
    ratingFirstPart = file_name[file_name.find("-")+1:file_name.find("_")] # between - and _
    ratingSecondPart = file_name[file_name.find("_")+1:file_name.find(".")] # between _ and .
    ratingComplete = ratingFirstPart + '.' + ratingSecondPart
    return float(ratingComplete)

def standardize_test_rating(valence):
    if(valence >= 5):
        return 1
    elif(valence <= 5):
        return 0

# create dictionary out of testset with a created id and standardizedRating and cleanedReview as values
def extract_external_testset_to_dict():
    for file in test_txt_files:
        file_name = extract_filename(file)
        test_id = extract_test_id(file_name)
        standardizedRating = standardize_test_rating(extract_test_rating(file_name))
        cleanedReview = read_first_line_txt(rel_path_to_testdata+file_name)
        
        external_testset_dict['id'].append(test_id)
        external_testset_dict['standardizedRating'].append(standardizedRating)
        external_testset_dict['cleanedReview'].append(cleanedReview)
        
        
# load lexicon to lexicon_dict
extract_lexical_features_to_dict()

# load external testfile to external_testset_dict
extract_external_testset_to_dict()

################## preparation overall_dict ##################
# load lexicon to lexicon_dict
print(f'Load lexicon...')
extract_lexical_features_to_dict()
# load review files
print(f'Load review files...')
txt_files = glob.glob(rel_path_to_trainingdata+"*.txt")

# generate overall_dict
for file in txt_files:
    filename = extract_filename(file)
    standardizedRating = standardize_ratings(extract_file_rating(filename))
    cleanedReview = clean_reviews(read_first_line_txt(rel_path_to_trainingdata+filename))

    overall_dict['id'].append(extract_file_id(filename))
    overall_dict['standardizedRating'].append(standardizedRating)
    overall_dict['cleanedReview'].append(cleanedReview)

# pickle overall_dict
with open(rel_path_to_additionaldata+"overall_dict_pickled.py", "wb") as outfile:
    pickle.dump(overall_dict, outfile)
print(f'Pickled overall_dict_pickled.py to {rel_path_to_additionaldata}overall_dict_pickled.py')

# pickle lexicon
with open(rel_path_to_additionaldata+"lexicon_pickled.py", "wb") as outfile:
    pickle.dump(lexicon_dict, outfile)
print(f'Pickled lexicon_pickled.py to {rel_path_to_additionaldata}lexicon_pickled.py')

# pickle external_testset_dict
with open(rel_path_to_additionaldata+"external_testset_dict_pickled.py", "wb") as outfile:
    pickle.dump(external_testset_dict, outfile)
print(f'Pickled external_testset_dict_pickled.py to {rel_path_to_additionaldata}external_testset_dict_pickled.py')

time_minutes = (time.time() - start_time)/60
print('Execution time in seconds: %s seconds' % (time.time() - start_time))
print('Execution time in minutes: %s minutes' % time_minutes)