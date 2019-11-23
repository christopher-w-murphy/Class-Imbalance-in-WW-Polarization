from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

def build_model(n_features, n_hidden_layers=2, n_neurons=150, loss_function='binary_crossentropy', final_bias='zeros'):
    model = Sequential()

    model.add(Dense(n_neurons, input_dim=n_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for _ in range(1, n_hidden_layers):
        model.add(Dense(n_neurons))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(1, bias_initializer=final_bias))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
    return model
