import numpy as np
import pandas as pd
import glob
import spacy
from spacy.tokens import Token

nlp = spacy.load('en_core_web_sm')
# elongated words
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



text = 'Hhello me Joeeel and I\'m soooooo .... excited!'
count = count_elongated_words(text)
print(count)

def negation_handling(review):
    doc = nlp(review)
    negationWordList = ['not', "\'nt"]
    negationContext = 0
    for i in range(len(doc)):
        if doc[i].text in negationWordList:
            j = i + 1
            negationContext += 1
            while doc[j].pos_ != "PUNCT" and j != (len(doc)-1):
                doc[j].set_extention("is_neg", getter=negation_getter)
                j += 1


def findNegationContexts(review):
    doc = nlp(review)
    negationCtx_list = []
    negationWordList = ['not', "n\'t"]
    for i in range(len(doc)):
        #print(doc[i].text)
        if doc[i].text in negationWordList:
            j = i + 1
            while doc[j].pos_ != "PUNCT" and j != (len(doc)-1):
                negationCtx_list.append(j) # append index of word between negWord and next punctuation
                j += 1
    print(negationCtx_list)

text2 = 'Hello I am not happy. Maybe I am happy though. I don\'t know yet bla, it is very hard to classify.'  
#negation_handling(text2)

#findNegationContexts(text2)

#beim nachschlagen der Valenz:
# lexiconbased features
# def extract_lexiconbased_features(review):
#     lexical_list = [] # [valence, arousal, dominance]
#     positiveWordCount = 0
#     negativeWordCount = 0
#     valence_values = []
#     arousal_values = []
#     dominance_values = []
#     doc = nlp(review)
#     for i in range(len(doc)):
#         token = doc[i]
#     # for token in doc: -> mit for i in range abgedeckt
#         text = token.text
#         valence = None
#         arousal = None
#         dominance = None
        
#         if text in lexicon_dict:
#             valence = lexicon_dict[text][0]
#             arousal = lexicon_dict[text][1]
#             dominance = lexicon_dict[text][2]
#             # Berechnung um Valenz um Mittelpunkt 5 zu drehen, falls Wort von Negation betroffen ist:
#             if i in negationWordList: # negation_list verfügbar machen für diese Methode
#                 if valence > 5:
#                     valence = valence - (valence - 5) # valence zu negativ umdrehen
#                 elif valence < 5:
#                     valence = valence + (5 - valence) # valence zu positiv umdrehen
#             if(float(valence) > 5):
#                 positiveWordCount += 1
#             elif(float(valence) < 5):
#                 negativeWordCount += 1
#         if(valence != None): # only append the values that could be found in lexicon
#             valence_values.append(float(valence))
#             arousal_values.append(float(arousal))
#             dominance_values.append(float(dominance))
#     valence_avg = sum(valence_values)/len(valence_values)
#     arousal_avg = sum(arousal_values)/len(arousal_values)
#     dominance_avg = sum(dominance_values)/len(dominance_values)
#     return (valence_avg, arousal_avg, dominance_avg, positiveWordCount, negativeWordCount)

text3 = "I haven't liked the tomatoes. I didn't hear you. Have you not seen him today?"

def getLemma(text):
    lemmalist = []
    doc = nlp(text)
    for token in doc:
        lemmalist.append(token.lemma_)
    print(lemmalist)

#getLemma(text3)


def findNegationContexts2(review):
    doc = nlp(review)
    tempNegCtx_list = []
    negationCtx_list = []
    negationWordList = ['not', "n\'t"]
    exceptionList1 = ["not only", "not just", "no question", "not to mention", "no wonder"]
    for i in range(len(doc)):
        #print(doc[i].text)
        if doc[i].text.lower() in negationWordList:
            j = i + 1
            while doc[j].pos_ != "PUNCT" and j != (len(doc)-1):
                tempNegCtx_list.append(j) # append index of word between negWord and next punctuation
                j += 1
            # check first exception
            bigram = f'{doc[i].text} {doc[i+1].text}'
            trigram = f"{doc[i].text} {doc[i+1].text} {doc[i+2].text}"
            print(f'bigram: {bigram} trigram: {trigram}')
            if bigram.lower() in exceptionList1 or trigram.lower() in exceptionList1:
                tempNegCtx_list = [] # Negation context does not apply
                continue
            # check second exception
            if doc[j].text == "?" and checkNegationException(doc, i) == True:
                tempNegCtx_list = [] # Negation context does not apply
            else:
                negationCtx_list.extend(tempNegCtx_list) # true negation context, extend negationCtx_list with previously found indexes
                tempNegCtx_list = []
    print(negationCtx_list)

def checkNegationException(doc, i):
    # if word in the first three places of document
    if i in (0, 1, 2):
        return True
    # if word in the first three places of a sentence
    elif doc[i-1].pos_ == "PUNCT":
        return True
    elif doc[i-2].pos_ == "PUNCT":
        return True
    elif doc[i-3].pos_ == "PUNCT":
        return True
    else:
        return False

text4 = "Have you not seen that movie? I do not like that. Don\'t you? Not only is it bad, it's also boring, not to mention a torture."
findNegationContexts2(text4)
