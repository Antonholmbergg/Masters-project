from os import times
from tensorflow import keras

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

    width = hp.Int(name='width', min_value=512, max_value=3072, step=256)
    depth = hp.Int(name='depth', min_value=1, max_value=6, step=1)
    #activation = hp.Choice(name='activation',  values=[keras.layers.LeakyReLU(), keras.activations.relu()])#values=['tanh', 'relu', 'selu'])
    activation = 'relu'
    #batch_norm = hp.Boolean(name='batch_norm')
    learning_rate = hp.Choice(name='learning_rate', values=[1e-3, 5e-4])
    
    
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


def get_multiple_output_model(width, depth, activation, input_shape=3, optimizer='adam', loss='mse', batch_norm=False):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    angles = keras.layers.Dense(256, activation=activation)(x)
    angles = keras.layers.Dense(128, activation=activation)(angles)
    angles = keras.layers.Dense(64, activation=activation)(angles)
    angles = keras.layers.Dense(4)(angles)
    
    times = keras.layers.Dense(128, activation=activation)(x)
    times = keras.layers.Dense(64, activation=activation)(times)
    times = keras.layers.Dense(32, activation=activation)(times)
    times = keras.layers.Dense(2)(times)
    
    lengths = keras.layers.Dense(128, activation=activation)(x)
    lengths = keras.layers.Dense(64, activation=activation)(lengths)
    lengths = keras.layers.Dense(32, activation=activation)(lengths)
    lengths = keras.layers.Dense(2)(lengths)
    
    outputs = [times, lengths, angles]
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def get_multiple_output_model_2(width, depth, activation, input_shape=3, optimizer='adam', loss='mse', batch_norm=False):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    recieve = keras.layers.Dense(128, activation=activation)(x)
    recieve = keras.layers.Dense(64, activation=activation)(recieve)
    recieve = keras.layers.Dense(32, activation=activation)(recieve)
    recieve = keras.layers.Dense(2)(recieve)
    
    launch = keras.layers.Dense(128, activation=activation)(x)
    launch = keras.layers.Dense(64, activation=activation)(launch)
    launch = keras.layers.Dense(32, activation=activation)(launch)
    launch = keras.layers.Dense(2)(launch)
    
    
    times = keras.layers.Dense(128, activation=activation)(x)
    times = keras.layers.Dense(64, activation=activation)(times)
    times = keras.layers.Dense(32, activation=activation)(times)
    times = keras.layers.Dense(2)(times)
    
    lengths = keras.layers.Dense(128, activation=activation)(x)
    lengths = keras.layers.Dense(64, activation=activation)(lengths)
    lengths = keras.layers.Dense(32, activation=activation)(lengths)
    lengths = keras.layers.Dense(2)(lengths)
    
    outputs = [times, lengths, launch, recieve]
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[100, 100, 10, 1])
    return model


def get_no_skip_model(width, activation, input_shape=3, output_shape=8, optimizer='adam', loss='mse'):
    inputs = keras.layers.Input(shape=(input_shape, ))
    x1 = keras.layers.Dense(width, activation=activation)(inputs)
    x2 = keras.layers.Dense(width, activation=activation)(x1)
    x3 = keras.layers.Dense(width, activation=activation)(x2)
    x4 = keras.layers.Dense(width, activation=activation)(x3)
    x5 = keras.layers.Dense(width, activation=activation)(x4)
    x6 = keras.layers.Dense(width, activation=activation)(x5)
    x7 = keras.layers.Dense(width, activation=activation)(x6)
    x8 = keras.layers.Dense(width, activation=activation)(x7)
    x9 = keras.layers.Dense(width, activation=activation)(x8)
    output = keras.layers.Dense(output_shape)(x9)
    
    model = keras.Model(inputs, output)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def get_skip_model(width, activation, input_shape=3, output_shape=8, optimizer='adam', loss='mse'):
    inputs = keras.layers.Input(shape=(input_shape, ))
    x1 = keras.layers.Dense(width, activation=activation)(inputs)
    x2 = keras.layers.Dense(width, activation=activation)(x1)
    x3 = keras.layers.Dense(width, activation=activation)(x2)
    x4 = keras.layers.Dense(width, activation=activation)(x3)
    x5 = keras.layers.Dense(width, activation=activation)(x4)
    x6 = keras.layers.Dense(width, activation=activation)(x5 + x4)
    x7 = keras.layers.Dense(width, activation=activation)(x6 + x3)
    x8 = keras.layers.Dense(width, activation=activation)(x7 + x2)
    x9 = keras.layers.Dense(width, activation=activation)(x8 + x1)
    output = keras.layers.Dense(output_shape)(x9)
    
    model = keras.Model(inputs, output)
    model.compile(optimizer=optimizer, loss=loss)
    return model