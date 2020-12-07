import glob
import shutil

#path_to_src = 'C:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/IMDb-Datenset_Training_Testing/test/pos/'
path_to_src = 'C:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/IMDb-Datenset_Training_Testing/test/neg/'
path_to_target = 'C:/Users/Joel/Share-NAS/Bachelorarbeit/Daten/IMDb-Datenset_Training_Testing/test/changedIDs/'


def extract_filename(file_string):
    return file_string.partition("\\")[2]

def extract_file_rating(file_name):
    return int(file_name[file_name.find("_")+1:file_name.find(".")])

#load files
txt_files = glob.glob(path_to_src+"*.txt")

#print(txt_files)
id_counter = 12500

#extract filename, id and rating
for id, file in enumerate(txt_files):
    old_filename= file.partition("\\")[2]
    rating = int(old_filename[old_filename.find("_")+1:old_filename.find(".")])
    new_filename = str(id_counter)+"_"+str(rating)
    shutil.move(path_to_src+old_filename, path_to_target+new_filename+'.txt')
    id_counter += 1