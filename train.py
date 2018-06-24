import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from copy import deepcopy

from nlp import model

input_length = 512
replace_n = 32
epochs = 5
batch_size = 4e2
train_lines = 9
valid_lines = 1
steps_per_epoch = 4 # 30e6 / batch_size

train_file = "/Users/paul/Downloads/WestburyLab.Wikipedia.Corpus/train.txt"
valid_file = "/Users/paul/Downloads/WestburyLab.Wikipedia.Corpus/valid.txt"
model = model(input_length)

I = np.identity(27)
def char_to_vector(char):
    return I[char]

def generate_lines(filename):
    with open(filename) as f:
        while f.seek(0) or True:
            for line in f:
                chars = np.array([c for c in line.encode('ascii', 'ignore').decode('ascii')], dtype='|S1').view(np.uint8)
                chars[chars < 97] += 32
                chars -= 96
                chars[chars == 224] = 0
                chars = chars[np.logical_and(chars >= 0, chars < 27)]
                data = np.apply_along_axis(char_to_vector, 0, chars)
                yield data

def noise(data):
    p = float(replace_n) / float(input_length)
    i = np.random.choice((True, False), input_length, p=(p, 1-p))
    n = np.sum(i)
    r = np.random.randint(0, 27, n)
    noisey = np.copy(data)
    noisey[i] = np.apply_along_axis(char_to_vector, 0, r)
    return noisey

def data_generator(filename):
    to_send = np.empty((0, 27))
    x_batch = []
    y_batch = []

    lines = generate_lines(filename)

    while True:
        while len(to_send) < input_length:
            to_send = np.append(to_send, next(lines), 0)

        data = to_send[:input_length]
        x_batch += [noise(data)]
        y_batch += [data]

        if len(y_batch) == batch_size:
            yield (np.array(x_batch), np.array(y_batch))
            x_batch = []
            y_batch = []

        to_send = to_send[input_length//2:]

def valid_generator():
    return data_generator(valid_file)

def train_generator():
    count = 0
    for batch in data_generator(train_file):
        yield batch
        count += 1
        print("{} / {} batches done".format(count, steps_per_epoch))

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=steps_per_epoch * (valid_lines / train_lines))
