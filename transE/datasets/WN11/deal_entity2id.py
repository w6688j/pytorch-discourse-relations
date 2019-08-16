import re

f = open('entity2id.txt', encoding='utf-8')
file_object = open('new_entity2id.txt', 'a+', encoding='utf-8')

for index, line in enumerate(f):
    if index == 0:
        file_object.write(line + '\n')
        continue
    else:
        line = line.strip('__')
        line = line.replace('\n', '')
        line_split = line.split("\t")

        word = line_split[0]
        replace = re.findall("[_]+[0-9]{0,}$", word)
        if len(replace):
            word = word.replace(replace[0], '')
            file_object.write(word + ' ' + line_split[1] + '\n')
            file_object.flush()

file_object.close()
f.close()
