import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.datasets.boston_housing import load_data
from sklearn.preprocessing import StandardScaler




if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    #scaler_x = StandardScaler().fit(X=x_train)
    #x_train = scaler_x.transform(X=x_train)
    #x_test = scaler_x.transform(X=x_test)
    scaler_y = StandardScaler().fit(X=y_train)
    y_train = scaler_y.transform(X=y_train)
    y_test = scaler_y.transform(X=y_test)
    
    model = Sequential()
    model.add(Input(shape=x_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation=tf.keras.activations.swish))
    model.add(Dense(units=5, activation=tf.keras.activations.swish))
    model.compile(optimizer='adam', loss='mse')
    model.summary(show_trainable=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=500)
    hist = model.fit(x=x_train, y=y_train, batch_size=32, validation_data=(x_test, y_test), verbose=True, epochs=100, callbacks=[callback])

    #print(f'Train loss: {np.mean((model.predict(x_train) - scaler_y.inverse_transform(y_train))**2)}')
    #print(f'Valid loss: {np.mean((model.predict(x_test) - scaler_y.inverse_transform(y_test))**2)}')
    print(f'Train loss: {np.mean((scaler_y.inverse_transform(model.predict(x_train)) - scaler_y.inverse_transform(y_train))**2)}')
    print(f'Valid loss: {np.mean((scaler_y.inverse_transform(model.predict(x_test)) - scaler_y.inverse_transform(y_test))**2)}')
    print(5) # model.predict(x=x_test)
