#https://www.youtube.com/watch?v=-GBgUy6ufUk
import spacy

def bigram(doc):
    #create a list for the 
    bigram_list = []
    #create a list that contains no punctuation
    sentence = []

    #parse through the document to add all tokens that are words to the sentence list
    for token in doc:
        if token.is_alpha:
            sentence.append(token.lemma_)

    #parse through the sentence while adding words in groups of two to the bigram_list
    for word in range(len(sentence) -1):
        firstWord = sentence[word]
        secondWord = sentence[word + 1]
        element = [firstWord, secondWord]
        bigram_list.append(element)

    return bigram_list


nlp = spacy.load('en_core_web_sm')

doc = nlp("I ordered cheese from the cheeser.")

result = bigram(doc)


#print("result type: ",type(result))
for element in result:
    #print("element type: ",type(element))
    for token in element:
        print(token, end=' ')
        #print("token type: ", type(token))
    print()