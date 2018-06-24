import tensorflow as tf
from keras.layers import Input, Conv1D, Activation, MaxPool1D, UpSampling1D
from layers import LayerNorm1D
from keras.layers.merge import Add
from keras.models import Model

def conv(filters, kernal_size, input):
    l = LayerNorm1D()(input)
    l = Activation('relu')(l)
    l = Conv1D(filters, kernal_size, padding='same')(l)
    return l

def model(input_length):
    input = Input(shape=(input_length, 27))

    l = Conv1D(64, 3, padding='same')(input)
    l = LayerNorm1D()(l)
    l = Activation('relu')(l)

    def f():
        a = Conv1D(128, 1, padding='same')(l)

        b = conv(64, 1, l)
        b = conv(64, 3, b)
        b = conv(128, 1, b)

        return Add()([a, b])
    l = f()

    def f():
        b = conv(64, 1, l)
        b = conv(64, 3, b)
        b = conv(128, 1, b)

        return Add()([l, b])
    l = f()

    def f():
        b = MaxPool1D()(l)
        def f():
            ba = conv(64, 1, b)
            ba = conv(64, 3, ba)
            ba = conv(128, 1, ba)

            ba = MaxPool1D()(ba)
            def f():
                baa = conv(64, 1, ba)
                baa = conv(64, 3, baa)
                baa = conv(128, 1, baa)

                baa = MaxPool1D()(baa)

                def f():
                    baaa = conv(64, 1, baa)
                    baaa = conv(64, 3, baaa)
                    baaa = conv(128, 1, baaa)
                    return Add()([baa, baaa])
                baa = f()

                baa = UpSampling1D()(baa)

                return Add()([ba, baa])
            ba = f()

            ba = UpSampling1D()(ba)

            return Add()([b, ba])
        b = f()

        b = UpSampling1D()(b)

        return Add()([l, b])
    l = f()

    l = conv(27, 1, l)
    l = LayerNorm1D()(l)

    l = Activation('sigmoid')(l)

    model = Model(input, l)

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model
