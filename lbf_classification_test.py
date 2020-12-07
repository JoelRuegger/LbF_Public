import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import preprocessing_pipeline_alt as lbf_t
import spacy

rel_path_to_additionaldata = 'additionalData/'
nlp = spacy.load('en_core_web_sm')

################## extraction shortstories ##################
# load shortStories from pickle
with open(rel_path_to_additionaldata+"external_testset_dict_pickled.py" , "rb") as infile:
    external_testset_dict_pickled = pickle.load(infile)

df_shortStories = pd.DataFrame(external_testset_dict_pickled)
shortStoriesLabels = df_shortStories['standardizedRating'].values
shortStories = df_shortStories['cleanedReview'].to_frame()

################## preparation for classification ##################
# extract short stories and corresponding labels
data_labels = shortStoriesLabels
# generate features of short stories
data = lbf_t.extendDataframe(df_shortStories)

################## classification ##################
with open('model_file.py', 'rb') as model_pickle, open('vectorizer.py', 'rb') as vec_pickle:
    vec = pickle.load(vec_pickle)
    fitted_data = vec.transform(data)
    classifier = pickle.load(model_pickle)
    predictions = classifier.predict(fitted_data)

    print(classification_report(data_labels, predictions))