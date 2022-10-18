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


def my_swish(x: NDArray, beta: float=2.0) -> NDArray:
    return x * tf.keras.activations.sigmoid(beta * x)

def my_lrelu(x: NDArray) -> NDArray:
    return tf.keras.activations.relu(x=x, alpha=0.1)

def max_soft(x1: NDArray, x2: NDArray, alpha: float=0.03) -> NDArray:
    return 0.5 * (x1 + x2 + K.sqrt(K.square(x1 - x2) + alpha))

def min_soft(x1: NDArray, x2: NDArray, alpha: float=0.03) -> NDArray:
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
        return 1.4 * self.dists.cdf(value=inputs) - 0.2



class RankActivation(Layer):
    def __init__(self, y_train: NDArray, activation: Callable[[NDArray], NDArray]):
        super().__init__()

        self.activation = activation

        self.num_feats = y_train.shape[1]
    
    def call(self, inputs: NDArray) -> NDArray:
        if inputs.shape[0] is None:
            return inputs

        a_s = Variable(
            initial_value=ones(shape=(self.num_feats,), dtype='float64'),
            shape=(self.num_feats,),
            trainable=True,
            name='A')
        b_s = Variable(
            initial_value=zeros(shape=(self.num_feats,), dtype='float64'),
            shape=(self.num_feats,),
            trainable=True,
            name='B')
        
        weights = Variable(
            initial_value=ones(shape=(self.num_feats * inputs.shape[1],), dtype='float64'),
            shape=(self.num_feats * inputs.shape[1],),
            trainable=True,
            name='C')
        
        return map_fn(
            elems=range(start=0, limit=inputs.shape[0]),
            fn_output_signature=tf.float64,
            fn=lambda idx: map_fn(
                elems=range(start=0, limit=self.num_feats),
                fn_output_signature=tf.float64,
                fn=lambda feat_idx: self.activation(
                    b_s[feat_idx] + a_s[feat_idx] * np.dot(
                        weights[feat_idx * inputs.shape[0]:(feat_idx + 1) * inputs.shape[0]],
                        inputs[idx,:]
                    )
                )
            )
        )





class RankOutput(Layer):
    def __init__(self, y_train: NDArray, stretch_sd: float=1.0, clip: bool=True) -> None:
        super(RankOutput, self).__init__()

        self.clip = clip
        
        self.dists = Normal(loc=y_train.T.mean(axis=1), scale=stretch_sd * y_train.T.std(axis=1), validate_args=True, allow_nan_stats=False)

    
    def call(self, inputs: NDArray):
        if self.clip:
            inputs = clip_by_value(t=inputs, clip_value_min=0.0, clip_value_max=1.0)
        return self.dists.quantile(value=inputs)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    ri = RankInput(x_train=x_train, stretch_sd=1.2)
    as_cdfs = ri(inputs=x_train[0:32,:]) # TODO: we gotta test whether this is correct, use some example data for which we know the answer

    ra = RankActivation(y_train=y_train, activation=my_swish)
    activated = ra(as_cdfs)


    ro = RankOutput(y_train=y_train, clip=True, stretch_sd=1.2)

    print(activated.shape)
    
    model = Sequential()
    model.add(Input(shape=x_train.shape[1]))
    model.add(ri)
    model.add(Dense(units=5, activation=tf.keras.activations.swish))
    #model.add(Dropout(rate=.2))
    model.add(Dense(units=5, activation=tf.keras.activations.swish))
    model.add(ra)
    model.add(TrainableSoftMinmax())
    model.add(ro)
    model.compile(optimizer='adam', loss='mse')
    model.summary(show_trainable=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=500)
    hist = model.fit(x=x_train, y=y_train, batch_size=32, validation_data=(x_test, y_test), verbose=True, epochs=200, callbacks=[callback])

    print(f'Train loss: {np.mean((model.predict(x_train) - y_train)**2)}')
    print(f'Valid loss: {np.mean((model.predict(x_test) - y_test)**2)}')




    temp = soft_minmax(x=as_cdfs)

    ro = RankOutput(y_train=y_train, stretch_sd=1.1, clip=True)
    ro_out = ro(inputs=temp)

    print(ro_out)

    #model.add(BatchNormalization())
    model.add(RankInput(x_train=x_train, stretch_sd=1.1))
    #model.add(Dense(units=5, activation=tf.keras.activations.swish))
    #model.add(Dense(units=y_train.shape[1], activation=tf.keras.activations.swish))
    model.add(TrainableSoftMinmax())
    model.add(RankOutput(y_train=y_train, clip=True, stretch_sd=1.1))
    #model.add(Dense(units=1, activation=tf.keras.activations.swish))
    model.summary(show_trainable=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model.fit(x=x_train, y=y_train, batch_size=32, validation_data=(x_test, y_test), verbose=True, epochs=500, callbacks=[callback])


    

    print(5)