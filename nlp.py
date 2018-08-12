import tensorflow as tf
from keras.layers import Input, Conv1D, Activation, MaxPool1D, UpSampling1D, ZeroPadding1D, Conv2DTranspose, BatchNormalization
from layers import LayerNorm1D
from keras.layers.merge import Add
from keras.models import Model
from math import ceil
import tensorflow as tf

def convF(filters, kernal_size, **kwargs):
    def a(input):
        norm = LayerNorm1D()(input) #BatchNormalization()(input)
        relu = Activation('relu')(norm)
        return Conv1D(filters, kernal_size, **kwargs)(relu)
    return a

def model1(input_length):
    input = Input(shape=(input_length, 27))

    def conv(filters, kernal_size, input):
        return convF(filters, kernal_size, padding='same', strides=1)(input)

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

def model2(input_length):
    input = Input(shape=(input_length, 27))

    def encode(input, filterss):
        def not_finished():
            filters = filterss[0]
            conv = convF(filters, 3, padding='valid', strides=2)(input)
            return encode(conv, filterss[1:])

        return input if len(filterss) == 0 else not_finished()

    filterss = (128, 256, 512, 1024)
    def count(x, i):
        return x if i == 0 else count(ceil(x / 2 - 1), i - 1)
    conv_length = count(input_length, len(filterss))
    encoded_features = 2048
    collapse = convF(encoded_features, conv_length, padding='valid', strides=1)
    encoded = collapse(encode(input, filterss))

    def decode(input, filterss):
        def not_finished():
            filters = filterss[0]
            up = UpSampling1D()(input)
            conv = convF(filters, 3, padding='same', strides=1)(up)
            return decode(conv, filterss[1:])

        return input if len(filterss) == 0 else not_finished()

    conv_features = filterss[-1]
    sideT = convF(conv_features * (conv_length + 1), encoded_features, padding='valid', strides=1, data_format='channels_first')(encoded)
    expand =  convF(conv_features, conv_features, padding='valid', strides=conv_features)(sideT)
    decoded = decode(expand, filterss[:-1][::-1] + (27,))

    output = Activation('sigmoid')(decoded) # (LayerNorm1D()(decoded))

    model = Model(input, output)

    for layer in model.layers:
        print("(%s) -> (%s)" % tuple([",".join(["" if n is None else str(n) for n in shape]) for shape in [layer.input_shape, layer.output_shape]]))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model