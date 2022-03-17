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


def get_multiple_output_model_skip(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths = keras.layers.Dense(1024, activation=activation)(x)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(2)(lengths)
    
    launch = keras.layers.Dense(1024, activation=activation)(x)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(2)(launch)
    
    recieve = keras.layers.Dense(1024, activation=activation)(x)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(2)(recieve)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[5, 1, 1, 1])
    return model


def get_multiple_output_model_skip_v2(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths = keras.layers.Dense(1024, activation=activation)(x)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(2)(lengths)
    
    launch = keras.layers.Dense(1024, activation=activation)(x)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(2)(launch)
    
    recieve = keras.layers.Dense(1024, activation=activation)(x)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(2)(recieve)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[10, 1, 1, 1])
    return model


def get_multiple_output_model_skip_v2_2(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths = keras.layers.Dense(1024, activation=activation)(x)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(1024, activation=activation)(lengths)
    lengths = keras.layers.Dense(2)(lengths)
    
    launch = keras.layers.Dense(1024, activation=activation)(x)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(1024, activation=activation)(launch)
    launch = keras.layers.Dense(2)(launch)
    
    recieve = keras.layers.Dense(1024, activation=activation)(x)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(1024, activation=activation)(recieve)
    recieve = keras.layers.Dense(2)(recieve)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[10, 1, 1, 2])
    return model



def get_multiple_output_model_skip_v3(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths1 = keras.layers.Dense(1024, activation=activation)(x)
    lengths2 = keras.layers.Dense(1024, activation=activation)(lengths1)
    lengths3 = keras.layers.Dense(1024, activation=activation)(lengths2 + lengths1)
    lengths4 = keras.layers.Dense(1024, activation=activation)(lengths3)
    lengths5 = keras.layers.Dense(1024, activation=activation)(lengths4 + lengths1)
    lengths6 = keras.layers.Dense(1024, activation=activation)(lengths5)
    lengths7 = keras.layers.Dense(1024, activation=activation)(lengths6 + lengths1)
    lengths8 = keras.layers.Dense(1024, activation=activation)(lengths7)
    lengths = keras.layers.Dense(2)(lengths8)
    
    launch1 = keras.layers.Dense(1024, activation=activation)(x)
    launch2 = keras.layers.Dense(1024, activation=activation)(launch1)
    launch3 = keras.layers.Dense(1024, activation=activation)(launch2 + launch1)
    launch4 = keras.layers.Dense(1024, activation=activation)(launch3)
    launch5 = keras.layers.Dense(1024, activation=activation)(launch4 + launch1)
    launch6 = keras.layers.Dense(1024, activation=activation)(launch5)
    launch7 = keras.layers.Dense(1024, activation=activation)(launch6 + launch1)
    launch8 = keras.layers.Dense(1024, activation=activation)(launch7)
    launch = keras.layers.Dense(2)(launch8)
    
    recieve1 = keras.layers.Dense(1024, activation=activation)(x)
    recieve2 = keras.layers.Dense(1024, activation=activation)(recieve1)
    recieve3 = keras.layers.Dense(1024, activation=activation)(recieve2 + recieve1)
    recieve4 = keras.layers.Dense(1024, activation=activation)(recieve3)
    recieve5 = keras.layers.Dense(1024, activation=activation)(recieve4 + recieve1)
    recieve6 = keras.layers.Dense(1024, activation=activation)(recieve5)
    recieve7 = keras.layers.Dense(1024, activation=activation)(recieve6 + recieve1)
    recieve8 = keras.layers.Dense(1024, activation=activation)(recieve7)
    recieve = keras.layers.Dense(2)(recieve8)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[10, 1, 1, 1])
    return model



def get_multiple_output_model_skip_v4(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths1 = keras.layers.Dense(1024, activation=activation)(x)
    lengths2 = keras.layers.Dense(1024, activation=activation)(lengths1)
    lengths3 = keras.layers.Dense(1024, activation=activation)(lengths2 + lengths1)
    lengths4 = keras.layers.Dense(1024, activation=activation)(lengths3)
    lengths5 = keras.layers.Dense(1024, activation=activation)(lengths4 + lengths1)
    lengths6 = keras.layers.Dense(1024, activation=activation)(lengths5)
    lengths = keras.layers.Dense(2)(lengths6)
    
    launch1 = keras.layers.Dense(1024, activation=activation)(x)
    launch2 = keras.layers.Dense(1024, activation=activation)(launch1)
    launch3 = keras.layers.Dense(1024, activation=activation)(launch2 + launch1)
    launch4 = keras.layers.Dense(1024, activation=activation)(launch3)
    launch5 = keras.layers.Dense(1024, activation=activation)(launch4 + launch1)
    launch6 = keras.layers.Dense(1024, activation=activation)(launch5)
    launch = keras.layers.Dense(2)(launch6)
    
    recieve1 = keras.layers.Dense(1024, activation=activation)(x)
    recieve2 = keras.layers.Dense(1024, activation=activation)(recieve1)
    recieve3 = keras.layers.Dense(1024, activation=activation)(recieve2 + recieve1)
    recieve4 = keras.layers.Dense(1024, activation=activation)(recieve3)
    recieve5 = keras.layers.Dense(1024, activation=activation)(recieve4 + recieve1)
    recieve6 = keras.layers.Dense(1024, activation=activation)(recieve5)
    recieve = keras.layers.Dense(2)(recieve6)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[10, 1, 1, 1])
    return model


def get_multiple_output_model_skip_v5(width, depth, activation, input_shape=3, optimizer='adam', loss='mse'):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)

    times1 = keras.layers.Dense(1024, activation=activation)(x)
    times2 = keras.layers.Dense(1024, activation=activation)(times1)
    times3 = keras.layers.Dense(1024, activation=activation)(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation)(times3)
    times5 = keras.layers.Dense(1024, activation=activation)(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation)(times5)
    times7 = keras.layers.Dense(1024, activation=activation)(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation)(times7)
    times = keras.layers.Dense(2)(times8)
    
    lengths1 = keras.layers.Dense(1024, activation=activation)(x)
    lengths2 = keras.layers.Dense(1024, activation=activation)(lengths1)
    lengths3 = keras.layers.Dense(1024, activation=activation)(lengths2 + lengths1)
    lengths4 = keras.layers.Dense(1024, activation=activation)(lengths3)
    lengths5 = keras.layers.Dense(1024, activation=activation)(lengths4 + lengths1)
    lengths6 = keras.layers.Dense(1024, activation=activation)(lengths5)
    lengths = keras.layers.Dense(2)(lengths6)
    
    launch1 = keras.layers.Dense(1024, activation=activation)(x)
    launch2 = keras.layers.Dense(1024, activation=activation)(launch1)
    launch3 = keras.layers.Dense(1024, activation=activation)(launch2)
    launch4 = keras.layers.Dense(1024, activation=activation)(launch3)
    launch5 = keras.layers.Dense(1024, activation=activation)(launch4)
    launch6 = keras.layers.Dense(1024, activation=activation)(launch5)
    launch = keras.layers.Dense(2)(launch6)
    
    recieve1 = keras.layers.Dense(1024, activation=activation)(x)
    recieve2 = keras.layers.Dense(1024, activation=activation)(recieve1)
    recieve3 = keras.layers.Dense(1024, activation=activation)(recieve2)
    recieve4 = keras.layers.Dense(1024, activation=activation)(recieve3)
    recieve5 = keras.layers.Dense(1024, activation=activation)(recieve4)
    recieve6 = keras.layers.Dense(1024, activation=activation)(recieve5)
    recieve = keras.layers.Dense(2)(recieve6)
    
    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[10, 1, 2, 4])
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


def get_skip_model_v2(width, activation, input_shape=3, output_shape=8, optimizer='adam', loss='mse'):
    inputs = keras.layers.Input(shape=(input_shape, ))
    x1 = keras.layers.Dense(width, activation=activation)(inputs)
    x2 = keras.layers.Dense(width, activation=activation)(x1)
    x3 = keras.layers.Dense(width, activation=activation)(x2)
    x4 = keras.layers.Dense(width, activation=activation)(x3 + x1)
    x5 = keras.layers.Dense(width, activation=activation)(x4)
    x6 = keras.layers.Dense(width, activation=activation)(x5 + x4)
    x7 = keras.layers.Dense(width, activation=activation)(x6)
    x8 = keras.layers.Dense(width, activation=activation)(x7 + x6)
    x9 = keras.layers.Dense(width, activation=activation)(x8)
    x10 = keras.layers.Dense(width, activation=activation)(x9 + x8)
    x11 = keras.layers.Dense(width, activation=activation)(x10)
    x12 = keras.layers.Dense(width, activation=activation)(x11 + x10)
    x13 = keras.layers.Dense(width, activation=activation)(x12)
    x14 = keras.layers.Dense(width, activation=activation)(x13 + x12)
    x15 = keras.layers.Dense(width, activation=activation)(x14)
    x16 = keras.layers.Dense(width, activation=activation)(x15 + x14)
    x17 = keras.layers.Dense(width, activation=activation)(x16)
    x18 = keras.layers.Dense(width, activation=activation)(x17 + x16)
    output = keras.layers.Dense(output_shape)(x18)
    
    model = keras.Model(inputs, output)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def get_simple_model_dropout(width, depth, activation, input_shape=3, output_shape=8, dropout_rate=0.2, optimizer='adam', loss='mse', batch_norm=False):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation)(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation)(x)
        x = keras.layers.Dropout(dropout_rate, noise_shape=(width,))(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(width/2, activation=activation)(x)
    x = keras.layers.Dropout(dropout_rate, noise_shape=(int(width/2),))(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(width/4, activation=activation)(x)
    #x = keras.layers.Dense(width/8, activation=activation)(x)
    outputs = keras.layers.Dense(output_shape)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model