import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

rel_path_to_trainingdata = 'partOfData/'
################## FEATURES ##################

with open(rel_path_to_trainingdata+"preprocessing_pickled.py" , "rb") as infile:
    preprocessed_data = pickle.load(infile)

review_strings = []
#sentence_strings = []
for key, value in preprocessed_data.items():
    review_strings.append(value[1])
    #sentence_strings.append(value[2][1])

# count pos tags and avg wordlength
pos_counts = {}
avgWordLength = {}
for key, value in preprocessed_data.items():
    print("...processing key "+str(key))
    noun_count = 0
    verb_count = 0
    adj_count = 0
    punct_count = 0
    for triple in value[2]:
        if triple[2] == "NOUN" or triple[2]== "PROPN" :
            noun_count += 1
        elif triple[2] == "VERB":
            verb_count += 1
        elif triple[2] == "ADJ":
            adj_count += 1
        elif triple[2] == "PUNCT":
            punct_count += 1

    pos_counts[key] = [noun_count, verb_count, adj_count, punct_count]  # {id: [noun_count, verb_count, adj_count, punct_count]}
    #count avg. wordlength without punctuation
    lengthWords = 0
    wordCounter = 0
    for posTriple in value[2]:
        if posTriple[2] != "PUNCT":
            lengthWords += len(posTriple[0])
            wordCounter += 1
    avgWordLength[key] = (lengthWords/wordCounter)

X = []
y = []

posList = []
classList = []
################## PREP FOR TRAINING WITH POS AND LABELS ##################
for label, pos in zip(preprocessed_data.values(), pos_counts.values()):
    X.append(pos)           #list of pos tags  [[31, 9, 12], [142, 66, 48]]    posList.append(pos)
    y.append(label[0])      #list of classes                                   classList.append(label[0])

#split data into 80% train and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#X = x_train     #00: [142, 66, 48] ... 40:
#y = y_train     #00: 1

print(X)
print(y)
        


################## TRAINING ##################
X_array = np.array(X_train) #features
print(X_array.shape) # how many items total, how many items per element if array in array
#x_reshaped = X.reshape(-1, 1)
y_array = np.array(y_train) #klassen
print(y_array.shape)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_array, y_array)


###################### TRAINING before splitting of train and testset
# X_array = np.array(X) #features
# print(X_array.shape) # how many items total, how many items per element if array in array
# #x_reshaped = X.reshape(-1, 1)
# y_array = np.array(y) #klassen
# print(y_array.shape)
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X_array, y_array)
# # Pipeline(steps=[('standardscaler', StandardScaler()),
# #                 ('svc', SVC(gamma='auto'))])

################## TESTING ##################
# np.set_printoptions(precision=2)
# confidence = clf.score(x_test, y_test)
# print('accuracy:', confidence)
X_testArray = np.array(X_train)
y_testArray = np.array(y_train)
print(clf.predict(X_test))
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print(clf.predict(x_test))
#print(clf.score(X_test, y_test))
#print(clf.predict([[64, 24, 22]]))