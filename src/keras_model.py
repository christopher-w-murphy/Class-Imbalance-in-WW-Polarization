from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def build_model(n_features, n_hidden_layers=2, n_neurons=150, loss_function='binary_crossentropy'):
    """
    deep, fully-connected neural network in Keras w/ Tensorflow backend
    outputs a binary classification probability
    input and hidden layers have 150 nodes (by default)
    are initialized w/ He
    have relu activation functions
    and are regularized using dropout w/ a 50% rate and batch normalization
    output layer has a sigmoid activation function
    the adam optimizer is used
    """
    model = Sequential()

    model.add(Dense(n_neurons, input_dim=n_features, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for _ in range(1, n_hidden_layers):
        model.add(Dense(n_neurons, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
    return model
