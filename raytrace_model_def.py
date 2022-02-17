from email.policy import default
from tensorflow import keras
import keras_tuner as kt


def get_simple_model(width, depth, activation, input_shape=3, output_shape=8, optimizer='adam', loss='mse', batch_norm=False):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Dense(output_shape)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def model_builder(hp):

    width = hp.Int(name='width', min_value=256, max_value=512, step=128)
    depth = hp.Int(name='depth', min_value=3, max_value=6, step=1)
    activation = hp.Choice(name='activation', values=['tanh', 'relu', 'sigmoid', 'selu'])
    #batch_norm = hp.Boolean(name='batch_norm')
    learning_rate = hp.Choice(name='learning_rate', values=[5e-3, 1e-3, 5e-4])
    
    
    model = get_simple_model(
        width=width,
        depth=depth,
        activation=activation#,
        #batch_norm=batch_norm
        )
    
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='mse'
        )
    
    return model