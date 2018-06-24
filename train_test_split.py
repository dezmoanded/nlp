data_file = "/Users/paul/Downloads/WestburyLab.Wikipedia.Corpus/WestburyLab.Wikipedia.Corpus.txt"
train_output = "/Users/paul/Downloads/WestburyLab.Wikipedia.Corpus/train.txt"
valid_output = "/Users/paul/Downloads/WestburyLab.Wikipedia.Corpus/valid.txt"

train_lines = 9
valid_lines = 1

with open(data_file) as f, open(train_output, mode='w') as t, open(valid_output, mode='w') as v:
    i = 0
    for line in f:
        if i == train_lines:
            v.write(line)
            i = 0
        else:
            t.write(line)
            i += 1