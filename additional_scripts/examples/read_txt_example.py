import glob

def read_first_line(file):
    with open(file, 'rt') as fd:
        first_line = fd.readline()
    return first_line

txt_files = glob.glob("test_txt/*.txt")
print(txt_files)

output_strings = map(read_first_line, txt_files)
review_list = list(output_strings)
print(review_list[0])

# for i in output_strings:
#     print(i)