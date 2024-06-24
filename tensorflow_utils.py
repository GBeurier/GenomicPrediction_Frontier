## Tensorflow models and callbacks
import math
import numpy as np

from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Activation,
    SpatialDropout1D,
    BatchNormalization,
    Flatten,
    Dropout,
    Input,
    GRU,
    LSTM,
    Bidirectional,
    MaxPool1D,
    AveragePooling1D,
    SeparableConv1D,
    Add,
    GlobalAveragePooling1D,
    GlobalMaxPool1D,
    Concatenate, concatenate,
    DepthwiseConv1D,
    Permute,
    MaxPooling1D,
    LayerNormalization,
    MultiHeadAttention,
    SeparableConv1D,
    ConvLSTM1D,
    LocallyConnected1D,
    Multiply,
    UpSampling1D,
    Lambda,
    Reshape
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


def cnn_softmax(nb_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=11, strides=1, activation='relu', input_shape=(nb_features, 1)))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=11, strides=3, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=9, strides=5, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=13, strides=5, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=5, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def cnn_bacon(nb_features):
    model = Sequential()
    model.add(Input(shape=(nb_features, 1)))
    model.add(SpatialDropout1D(0.08))
    model.add(Conv1D(filters=8, kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


# Optimized callback to manage model checkpoints in memory
class Auto_Save(Callback):
    best_weights = []

    def __init__(self, model_name, verbose = 0):
        super(Auto_Save, self).__init__()
        self.model_name = model_name
        self.best = np.Inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best):
            if self.verbose > 0:
                print("epoch", str(epoch).zfill(5), "loss", "{:.6f}".format(current_loss), "{:.6f}".format(self.best), " " * 10)
            self.best = current_loss
            self.best_epoch = epoch
            Auto_Save.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.verbose > 1:
            print("Saved best {0:6.4f} at epoch".format(self.best), self.best_epoch)
        self.model.set_weights(Auto_Save.best_weights)
        self.model.save_weights("models/" + self.model_name + ".hdf5")
        # self.model.save(self.model_name + ".h5")
        with open("models/" + self.model_name + "_summary.txt", "w") as f:
            with redirect_stdout(f):
                self.model.summary()


def scale_fn(x):
    # return 1. ** x
    return 1 / (2.0 ** (x - 1))

# Cyclical learning rate function
def clr(epoch):
    cycle_params = {
        "MIN_LR": 0.00001,
        "MAX_LR": 0.01,
        "CYCLE_LENGTH": 128,
    }
    MIN_LR, MAX_LR, CYCLE_LENGTH = (
        cycle_params["MIN_LR"],
        cycle_params["MAX_LR"],
        cycle_params["CYCLE_LENGTH"],
    )
    initial_learning_rate = MIN_LR
    maximal_learning_rate = MAX_LR
    step_size = CYCLE_LENGTH
    step_as_dtype = float(epoch)
    cycle = math.floor(1 + step_as_dtype / (2 * step_size))
    x = abs(step_as_dtype / step_size - 2 * cycle + 1)
    mode_step = cycle  # if scale_mode == "cycle" else step
    return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1 - x)) * scale_fn(mode_step)

