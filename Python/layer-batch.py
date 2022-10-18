import numpy as np
from typing import Callable
from keras import backend as K
from keras.models import Sequential
from keras.layers import Layer, Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow_probability.python.distributions import Normal
from keras.datasets.boston_housing import load_data
from nptyping import NDArray
from tensorflow import Variable, clip_by_value, ones, zeros, map_fn, range, reshape, matmul, tensordot, reduce_sum, multiply, transpose
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
tf.keras.backend.set_floatx('float64')




def my_swish(x: NDArray, beta: float=.1) -> NDArray:
    return x * tf.keras.activations.sigmoid(beta * x)

def max_soft(x1: NDArray, x2: NDArray, alpha: float=0.025) -> NDArray:
    return 0.5 * (x1 + x2 + K.sqrt(K.square(x1 - x2) + alpha))

def min_soft(x1: NDArray, x2: NDArray, alpha: float=0.025) -> NDArray:
    return 0.5 * (x1 + x2 - K.sqrt(K.square(x1 - x2) + alpha))

def soft_minmax(x: NDArray) -> NDArray:
    return min_soft(x1=1.0, x2=max_soft(x1=0.0, x2=x))




class TrainableSoftMinmax(Layer):
    def __init__(self):
        super(TrainableSoftMinmax, self).__init__()

        self.a = Variable(initial_value=1.0, dtype='float64', trainable=True, name='G')
        self.b = Variable(initial_value=0.0, dtype='float64', trainable=True, name='H')
    

    def call(self, inputs: NDArray) -> NDArray:
        return soft_minmax(self.b + self.a * inputs)



class RankInput(Layer):
    def __init__(self, x_train: NDArray, stretch_sd: float=1.0) -> None:
        super(RankInput, self).__init__()

        # axis=1 is the mean for all rows, but we wnat columns -> transpose!
        self.dists = Normal(loc=x_train.T.mean(axis=1), scale=stretch_sd * x_train.T.std(axis=1), validate_args=True, allow_nan_stats=False)
    

    def call(self, inputs: NDArray):
        return self.dists.cdf(value=inputs)

from keras.layers import Layer
import tensorflow as tf

class RankBatchNormalization(Layer):
    def __init__(self):
        super(RankBatchNormalization, self).__init__()

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )

        self.gamma = self.add_weight(
            shape=(input_shape[-1]),
            initializer="ones",
            trainable=True,
        )

        self.moving_mean = self.add_weight(
            shape=(input_shape[-1]),
            initializer=tf.initializers.zeros,
            trainable=False)

        self.moving_std = self.add_weight(
            shape=(input_shape[-1]),
            initializer=tf.initializers.ones,
            trainable=False)

    def get_moving_average(self, statistic, new_value):
        momentum = 0.9
        new_value = statistic * momentum + new_value * (1 - momentum)
        return statistic.assign(new_value)
    
    def rank_transform(self, x, x_mean, x_sd):
        dist = Normal(loc=x_mean, scale=x_sd + 1e-6, validate_args=True, allow_nan_stats=False)
        return dist.cdf(value=x)

    def call(self, inputs, training):
        if training:
            assert len(inputs.shape) in (2, 4)
            if len(inputs.shape) > 2:
                axes = [0, 1, 2]
            else:
                axes = [0]
            mean, var = tf.nn.moments(inputs, axes=axes, keepdims=False)
            std = tf.sqrt(var)
            self.moving_mean.assign(self.get_moving_average(self.moving_mean, mean))
            self.moving_std.assign(self.get_moving_average(self.moving_std, std))
        else:
            mean, std = self.moving_mean, tf.sqrt(self.moving_std)

        return self.beta + self.gamma * self.rank_transform(inputs, mean, std)

# class RankBatchNormalization(Layer):
#     def __init__(self):
#         super(RankBatchNormalization, self).__init__()
#         #self.means: list[NDArray]  = []
#         #self.sds: list[NDArray] = []

#         self.means = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
#         self.sds = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    
#     @tf.function
#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         inputs_t = transpose(inputs)
#         mean_ = tf.math.reduce_mean(input_tensor=inputs_t, axis=1)
#         self.means.write(index=self.means.size(), value=mean_)

#         sd_ = tf.math.reduce_std(input_tensor=inputs_t, axis=1)
#         self.sds.write(index=self.sds.size(), value=sd_)

#         loc = tf.math.reduce_mean(input_tensor=transpose(self.means), axis=1)
#         sca = tf.math.reduce_std(input_tensor=transpose(self.sds), axis=1)

#         dist = Normal(loc=loc, scale=sca, allow_nan_stats=False, validate_args=True)
#         return dist.cdf(value=inputs)

class RankInput(Layer):
    def __init__(self, x_train: NDArray, stretch_sd: float=1.0) -> None:
        super(RankInput, self).__init__()

        # axis=1 is the mean for all rows, but we wnat columns -> transpose!
        self.dists = Normal(loc=x_train.T.mean(axis=1), scale=stretch_sd * x_train.T.std(axis=1), validate_args=True, allow_nan_stats=False)
    

    def call(self, inputs: NDArray):
        return self.dists.cdf(value=inputs)




if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    scaler_y = StandardScaler().fit(X=y_train)
    y_train = scaler_y.transform(X=y_train)
    y_test = scaler_y.transform(X=y_test)
    
    
    
    model = Sequential()
    model.add(Input(shape=x_train.shape[1]))
    #model.add(RankBatchNormalization())
    model.add(RankInput(x_train=x_train))
    model.add(Dense(units=20, activation=tf.keras.activations.swish))
    model.add(Dense(units=1, activation=tf.keras.activations.swish))
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.summary(show_trainable=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    hist = model.fit(x=x_train, y=y_train, batch_size=32, validation_data=(x_test, y_test), verbose=True, epochs=5000, callbacks=[callback])


    print(f'Train loss: {np.mean((scaler_y.inverse_transform(model.predict(x_train)) - scaler_y.inverse_transform(y_train))**2)}')
    print(f'Valid loss: {np.mean((scaler_y.inverse_transform(model.predict(x_test)) - scaler_y.inverse_transform(y_test))**2)}')

    print(5)