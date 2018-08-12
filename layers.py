from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer
import keras
import numpy as np
from keras import callbacks as cbks
from keras.utils.generic_utils import Progbar

class LayerNorm1D(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=keras.initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True,)

        super().build(input_shape)

    def call(self, x):
        # mean = K.mean(x, axis=-1, keepdims=True)
        # std = K.std(x, axis=-1, keepdims=True)
        mean = K.mean(x, keepdims=True)
        std = K.std(x, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

def calc_stats(W):
    return np.linalg.norm(W, 2), np.mean(W), np.std(W)

class MyDebugWeights(Callback):

    def __init__(self):
        super(MyDebugWeights, self).__init__()
        self.weights = []
        self.tf_session = K.get_session()

    def on_batch_end(self, epoch, logs=None):
        pass
        # for layer in self.model.layers:
        #     name = layer.name
        #     for i, w in enumerate(layer.weights):
        #         w_value = w.eval(session=self.tf_session)
        #         w_norm, w_mean, w_std = calc_stats(np.reshape(w_value, -1))
        #         self.weights.append((epoch, "{:s}/W_{:d}".format(name, i),
        #                              w_norm, w_mean, w_std))
        # for e, k, n, m, s in self.weights:
        #     print("{:3d} {:20s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m, s))

    # def on_batch_end(self, logs=None):
    #     for e, k, n, m, s in self.weights:
    #         print("{:3d} {:20s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m, s))

def callbacks(model, callbacks, params):
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    _callbacks.append(
        cbks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]
    callbacks.set_params({
        **params,
        'metrics': callback_metrics,
    })
    return callbacks