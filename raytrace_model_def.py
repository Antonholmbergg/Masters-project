from tensorflow import keras
import keras_tuner as kt


def get_simple_model(width, depth, activation, input_shape=3, output_shape=8, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    outputs = keras.layers.Dense(output_shape)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def model_builder(hp):

    width = hp.Int(name='width', min_value=64, max_value=128, step=64)
    depth = hp.Int(name='depth', min_value=5, max_value=6, step=1)
    activation = hp.Choice(name='activation', values=['tanh', 'relu'])#, 'sigmoid', 'selu'])
    
    model = get_simple_model(
        width=width,
        depth=depth,
        activation=activation
        )
    
    model.compile(
        optimizer='adam',
        loss='mse'
        )
    
    return model