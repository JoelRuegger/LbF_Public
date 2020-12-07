import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

################## FEATURES ##################

with open("test_txt/preprocessing_pickled.py" , "rb") as infile:
    preprocessed_data = pickle.load(infile)



review_strings = []
for key, value in preprocessed_data.items():
    review_strings.append(value[1])

pos_counts = {}
lexical_features = {}
for key, value in preprocessed_data.items():
    print("...processing key "+str(key))
    noun_count = 0
    verb_count = 0
    adj_count = 0
    #count pos-tags
    for triple in value[2]:
        if triple[2] == "NOUN" or triple[2]== "PROPN" :
            noun_count += 1
        elif triple[2] == "VERB":
            verb_count += 1
        elif triple[2] == "ADJ":
            adj_count += 1
    pos_counts[key] = [noun_count, verb_count, adj_count]

    for triple in value[3]:
        valence = triple[0]
        arousal = triple[1]
        dominance = triple[2]
    lexical_features[key] = [valence, arousal, dominance]

X = []
y = []
for label, lex in zip(preprocessed_data.values(), lexical_features.values()):
     X.append(lex) #list of pos tags
     y.append(label[0]) #list of classes

print(X)
print(y)
        


################## TRAINING ##################

X_array = np.array(X) #features
x_reshaped = X_array.reshape(-1, 1)
y_array = np.array(y) #klassen
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_reshaped, y_array)
# # Pipeline(steps=[('standardscaler', StandardScaler()),
# #                 ('svc', SVC(gamma='auto'))])
testArray = np.array([-0.2])
testArrayR = testArray.reshape(-1, 1)
print(clf.predict(testArrayR))