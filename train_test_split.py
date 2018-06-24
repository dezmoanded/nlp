from sys import argv

data_file = argv[1]
train_output = argv[2]
valid_output = argv[3]

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