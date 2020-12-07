#script to read csv and write csv to new_lexicon.csv line per line with delimiter "tab" ('\t')

import csv

#open specified csv, csv_reader can have any variablename
with open('testList.csv','r') as f:
    csv_reader = csv.reader(f)

    
    #open new file for writing, csv_writer can have any variablename
    with open('new_lexicon.csv', 'w') as new_file:
        #fieldnames = ['Words', 'Valence-mean-sum']
        csv_writer = csv.writer(new_file, delimiter='\t')

        #csv_writer.writerow(fieldnames)
        #omits the title row
        next(csv_reader)

        for row in csv_reader:
            csv_writer.writerow(row)