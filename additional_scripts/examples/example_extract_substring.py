my_string = "test_txt\\2_9.txt"

filename = my_string.partition("\\")[2]

file_id = filename[:filename.find("_")]  #from start to one before _ [start:end]
file_rating = filename[filename.find("_")+1:filename.find(".")]
print(file_id)
print(file_rating)

#file_id = filename("_")[0]
#file_rating = filename("_")[2]

#print(filename)



