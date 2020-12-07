#script to read valences from a csv and save them in a list
import csv



with open('testList.csv','r') as f:
    lexicon = list(csv.reader(f))
    
    wordIndex = 1
    valenceIndex = 2
    arousalIndex = 5
    dominanceIndex = 8
    searchTerm = "abdomen"
    my_dict = {}

    #create dictionary out of lexicon with word as key and valence, arousal and dominance as values
    for id, row in enumerate(lexicon):
        #my_dict[key] = "value" https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/
        my_dict[lexicon[id][wordIndex]] = (lexicon[id][valenceIndex], lexicon[id][arousalIndex], lexicon[id][dominanceIndex])
        

    print(my_dict)

    if searchTerm in my_dict:
        #TODO print titles
        print(searchTerm+" "+my_dict[searchTerm][0]+" "+my_dict[searchTerm][1]+" "+my_dict[searchTerm][2])
