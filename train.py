import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import keras.backend as K
from sklearn.model_selection import train_test_split
from copy import deepcopy
from params import train_file, valid_file
from layers import MyDebugWeights, callbacks

from nlp import model2

input_length = 512
replace_n = 32
epochs = 15
batch_size = 200
train_lines = 9
valid_lines = 1
steps_per_epoch = 2000 # 30e6 / batch_size

model = model2(input_length)

I = np.identity(27)
def char_to_vector(char):
    return I[char]

def to_chars(data):
    b = np.zeros_like(data)
    argm = data.argmax(2)
    def app(x):
        return argm == x
    b[np.array([app(x) for x in np.arange(27)]).swapaxes(0, 1).swapaxes(1, 2)] = 1
    chars = np.where(b == 1)[-1].reshape(b.shape[:2])
    chars += 96
    chars[chars == 96] = 32
    return chars.astype('uint8').view('c').astype('unicode')

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

_callbacks = [MyDebugWeights(),
             EarlyStopping(monitor='val_loss',
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
             TensorBoard(log_dir='logs', histogram_freq=1, write_grads=True)]
params = {
    'epochs': epochs,
    'steps': steps_per_epoch,
    'verbose': True,
    'do_validation': True,
}
callbacks = callbacks(model, _callbacks, params)

# model.fit_generator(generator=train_generator(),
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=epochs,
#                     verbose=2,
#                     callbacks=callbacks,
#                     validation_data=valid_generator(),
#                     validation_steps=steps_per_epoch * (valid_lines / train_lines))

def valid_data(val_x, val_y):
    val_x, val_y, val_sample_weights = model._standardize_user_data(
        val_x, val_y, None)
    val_data = val_x + val_y + val_sample_weights

    model.evaluate(val_x, val_y, batch_size=batch_size)

    t = model.predict(val_x, batch_size=batch_size)
    t_chars = to_chars(t)
    p = to_chars(val_x[0])
    print("\n".join(["".join(line) for line in t_chars[:5]]))
    print("\n")
    print("\n".join(["".join(line) for line in p[:5]]))

    if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                    int):
        val_data += [0.]
    return val_data

callbacks.on_train_begin()
train = train_generator()
validate = valid_generator()
for epoch in range(epochs):
    callbacks.on_epoch_begin(epoch)
    for batch in range(steps_per_epoch):
        x, y = next(train)
        callbacks.on_batch_begin(batch)
        stats = model.train_on_batch(x, y)
        callbacks.on_batch_end(batch)
        batch += 1
    x, y = next(validate)
    val_data = valid_data(x, y)
    for cbk in callbacks:
        cbk.validation_data = val_data
    print("{} {}".format(epoch, val_data))
    callbacks.on_epoch_end(epoch)
callbacks.on_train_end()
