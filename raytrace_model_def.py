"""Defines the raytracing models. Many different versions, mainly the resnet-like model which has the best performance.
    Tested with tensorflow 2.2
Returns:
    _type_: _description_
"""
from tensorflow import keras


class LossHistory(keras.callbacks.Callback):
    """A callback that saves the loss history

    Args:
        keras (_type_): _description_
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        if batch % 100 == 0:
            logs['batch'] = batch
            self.losses.append(logs)


class ResNet_Fc_block(keras.layers.Layer):
    """Defines a ResNet-like block.

    Args:
        keras (_type_): _description_
    """
    def __init__(self, width, act=keras.layers.LeakyReLU(), kernel_init='he_normal'):
        """Initializes the block

        Args:
            width (int): the width of the layers in the block
            act (activation function, optional): As a function and not the string name. Defaults to keras.layers.LeakyReLU().
            kernel_init (, optional): Type of kernek initializer, depends on the activation function used. Defaults to 'he_normal'.
        """
        super().__init__()
        self.width = width
        self.act = act
        self.kernel_init = kernel_init

    def build(self, input_shape):
        self.layer1 = keras.layers.Dense(
            self.width, kernel_initializer=self.kernel_init
            )
        self.layer2 = keras.layers.Dense(
            self.width, kernel_initializer=self.kernel_init
            )
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'width': self.width,
            'act': self.act,
            'kernel_init': self.kernel_init,
        })
        return config

    def call(self, inputs):
        x = self.ln1(inputs)
        x = self.layer1(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.layer2(x)
        x = self.act(x)
        return x + inputs


class ResNet_Fc(keras.layers.Layer):
    """Connects many resnet-like blocks in sequence

    Args:
        keras (_type_): _description_
    """
    def __init__(self, depth, width, act=keras.layers.LeakyReLU(),
                 kernel_init='he_normal'):
        """Initializes the network

        Args:
            depth (int): number of blocks (number of laers is double)
            width (int): width of the layers in the blocks
            act (activation finction, optional): as function and not as the string name. Defaults to keras.layers.LeakyReLU().
            kernel_init (, optional): kernel initialiser. Defaults to 'he_normal'.
        """
        super().__init__()
        self.depth = depth
        self.width = width
        self.act = act
        self.kernel_init = kernel_init

    def build(self, input_shape):
        self.first_layer = keras.layers.Dense(self.width, kernel_initializer=self.kernel_init)
        for i in range(self.depth):
            setattr(self, f'resnet_fc_block_{i}', ResNet_Fc_block(width=self.width, act=self.act, kernel_init=self.kernel_init))

        self.last_layer = keras.layers.Dense(self.width, kernel_initializer=self.kernel_init)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'width': self.width,
            'act': self.act,
            'kernel_init': self.kernel_init,
        })
        return config

    def call(self, inputs):
        x = self.first_layer(inputs)
        x = self.act(x)
        for i in range(self.depth):
            x = getattr(self, f'resnet_fc_block_{i}')(x)

        x = self.last_layer(x)
        return self.act(x)


def get_resnet_model_lrelu(width, depth, activation, input_shape=3,
                           optimizer='adam', loss='mse', kernel_init='he_normal'):
    """Builds a model with four outputs using the resnet network. 
    Currently with manual changes to the depths and widths to have different number 
    of parameters for the different outputs. Should be changed so it is defined when
    the function is called.

    Args:
        width (int): the width of the layers
        depth (int): the number of blocks to use in the outputs
        activation (activation function): activation function to use, as function and not as the string name.
        input_shape (int, optional): thae shape of the input. Defaults to 3.
        optimizer (str, optional): the name of the optimizer. Defaults to 'adam'.
        loss (str, optional): the name of the loss function. Defaults to 'mse'.
        kernel_init (, optional): the kernel initializer. Defaults to 'he_normal'.

    Returns:
        _type_: compiled keras model
    """

    inputs = keras.Input(shape=(input_shape, ))

    x = keras.layers.Dense(width, activation=activation,
                           kernel_initializer=kernel_init)(inputs)

    times = ResNet_Fc(depth+4, width+20, act=activation, kernel_init=kernel_init)(x)
    times = keras.layers.Dense(2, activation='sigmoid')(times)
    lengths = ResNet_Fc(depth-2, width, act=activation, kernel_init=kernel_init)(x)
    lengths = keras.layers.Dense(2, activation='sigmoid')(lengths)
    launch = ResNet_Fc(depth-2, width, act=activation, kernel_init=kernel_init)(x)
    launch = keras.layers.Dense(2, activation='sigmoid')(launch)
    receive = ResNet_Fc(depth-2, width, act=activation, kernel_init=kernel_init)(x)
    receive = keras.layers.Dense(2, activation='sigmoid')(receive)

    outputs = [times, lengths, launch, receive]
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=[1000, 10, 1, 1]
        )
    return model


class ResNet_Fc_block_new(keras.layers.Layer):
    """Same as the normal ResNet block but with the layer norm optional

    Args:
        keras (_type_): _description_
    """
    def __init__(self, width, act, kernel_init, ln):
        """initializes the block

        Args:
            width (int): the width of the dense layers
            act (activation function): as function or string name
            kernel_init (): kernel initializer as string or as initializer
            ln (bool): whether to use layer norm or not
        """
        super().__init__()
        self.width = width
        self.act = act
        self.kernel_init = kernel_init
        self.ln = ln

    def build(self, input_shape):
        self.layer1 = keras.layers.Dense(
            self.width,
            activation=self.act,
            kernel_initializer=self.kernel_init
            )
        self.layer2 = keras.layers.Dense(
            self.width,
            activation=self.act,
            kernel_initializer=self.kernel_init
            )
        if self.ln:
            self.ln1 = keras.layers.LayerNormalization()
            self.ln2 = keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'width': self.width,
            'act': self.act,
            'kernel_init': self.kernel_init,
            'layer_norm': self.ln,
        })
        return config

    def call(self, inputs):
        if self.ln:
            x = self.ln1(inputs)
            x = self.layer1(x)
            x = self.ln2(x)
            x = self.layer2(x)
        else:
            x = self.layer1(inputs)
            x = self.layer2(x)
        return x + inputs


class ResNet_Fc_new(keras.layers.Layer):
    """Same as ResNet_Fc exept layer norm is optional and activation function is
    defined in the dense layer instead of as its own layer.

    Args:
        keras (_type_): _description_
    """
    def __init__(self, depth, width, act, kernel_init, ln):
        """initialises the resnet-like structure

        Args:
            depth (int): number of blocks to use
            width (int): the number of nodes in the hidden layers
            act (activation unction): the activation function in the dens layer
            kernel_init (): kearnel initializer as string od initialiser
            ln (bool): wether to use layer norm or not
        """
        super().__init__()
        self.depth = depth
        self.width = width
        self.act = act
        self.kernel_init = kernel_init
        self.ln = ln

    def build(self, input_shape):
        self.first_layer = keras.layers.Dense(self.width, activation=self.act, kernel_initializer=self.kernel_init)
        for i in range(self.depth):
            setattr(self, f'resnet_fc_block_{i}', ResNet_Fc_block_new(width=self.width, act=self.act, kernel_init=self.kernel_init, ln=self.ln))

        self.last_layer = keras.layers.Dense(self.width, activation=self.act, kernel_initializer=self.kernel_init)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'width': self.width,
            'act': self.act,
            'kernel_init': self.kernel_init,
            'layer_norm': self.ln
        })
        return config

    def call(self, inputs):
        x = self.first_layer(inputs)
        for i in range(self.depth):
            x = getattr(self, f'resnet_fc_block_{i}')(x)

        return self.last_layer(x)


def get_resnet_model_new(width, depth, activation, kernel_init, ln,
                         input_shape=3, optimizer='adam', loss='mse'):
    """Defines and compiles a resnet-like model. Change in depths and widths are set manully
    this sould be changed to be defined in the function call.

    Args:
        width (int): number of nods in each hidden layer
        depth (int): the number of resnet-blocks
        activation (activation): function or string name
        kernel_init (_type_): kernel initializer as string or initializer object
        ln (bool): whether to use layer norm or not
        input_shape (int, optional): _description_. Defaults to 3.
        optimizer (str, optional): name of the optimizer. Defaults to 'adam'.
        loss (str, optional): name of the loss function. Defaults to 'mse'.

    Returns:
        _type_: compiled keras model
    """
    kwargs = {'act': activation, 'kernel_init': kernel_init, 'ln': ln}
    inputs = keras.Input(shape=(input_shape, ))

    x = keras.layers.Dense(width, activation=activation,
                           kernel_initializer=kernel_init)(inputs)

    times = ResNet_Fc_new(depth+2, (width + 80), **kwargs)(x)
    times = keras.layers.Dense(2, activation='sigmoid')(times)
    lengths = ResNet_Fc_new(depth, width, **kwargs)(x)
    lengths = keras.layers.Dense(2, activation='sigmoid')(lengths)
    launch = ResNet_Fc_new(depth, width, **kwargs)(x)
    launch = keras.layers.Dense(2, activation='sigmoid')(launch)
    receive = ResNet_Fc_new(depth, width, **kwargs)(x)
    receive = keras.layers.Dense(2, activation='sigmoid')(receive)

    outputs = [times, lengths, launch, receive]
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=[1000, 10, 1, 1]
        )
    return model


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# After this are a bunch of old models that perform worse but are kept in for completeness
# Some use the keras tuner
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


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
    # activation = hp.Choice(name='activation',  values=[keras.layers.LeakyReLU(), keras.activations.relu()])#values=['tanh', 'relu', 'selu'])
    activation = 'relu'
    # batch_norm = hp.Boolean(name='batch_norm')
    learning_rate = hp.Choice(name='learning_rate', values=[1e-3, 5e-4])

    model = get_simple_model(
        width=width,
        depth=depth,
        activation=activation  # ,
        # batch_norm=batch_norm
        )

    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss='mse'
        )

    return model


def get_multiple_output_model(width, depth, activation, input_shape=3,
                              optimizer='adam', loss='mse', batch_norm=False):

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


def get_multiple_output_model_skip(width, depth, activation, input_shape=3,
                                   optimizer='adam', loss='mse'):

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


def get_multiple_output_model_skip_v2(width, depth, activation, input_shape=3,
                                      optimizer='adam', loss='mse'):

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


def get_multiple_output_model_skip_v2_2(
        width, depth, activation, input_shape=3, optimizer='adam', loss='mse'
        ):

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


def get_multiple_output_model_skip_sig_selu(
        width, depth, activation, input_shape=3, optimizer='adam', loss='mse'
        ):

    inputs = keras.Input(shape=(input_shape, ))
    x = keras.layers.Dense(width, activation=activation, kernel_initializer='he_normal')(inputs)
    for i in range(1, depth):
        x = keras.layers.Dense(width, activation=activation, kernel_initializer='he_normal')(x)

    times1 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(x)
    times2 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times1)
    times3 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times2 + times1)
    times4 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times3)
    times5 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times4 + times1)
    times6 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times5)
    times7 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times6 + times1)
    times8 = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(times7)
    times = keras.layers.Dense(2, activation='sigmoid')(times8)

    lengths = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(x)
    lengths = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(lengths)
    lengths = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(lengths)
    lengths = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(lengths)
    lengths = keras.layers.Dense(2, activation='sigmoid')(lengths)

    launch = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(x)
    launch = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(launch)
    launch = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(launch)
    launch = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(launch)
    launch = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(launch)
    launch = keras.layers.Dense(2, activation='sigmoid')(launch)

    recieve = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(x)
    recieve = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(recieve)
    recieve = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(recieve)
    recieve = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(recieve)
    recieve = keras.layers.Dense(1024, activation=activation, kernel_initializer='he_normal')(recieve)
    recieve = keras.layers.Dense(2, activation='sigmoid')(recieve)

    outputs = [times, lengths, launch, recieve]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[100, 1, 1, 1])
    return model


def get_multiple_output_model_skip_v3(width, depth, activation, input_shape=3,
                                      optimizer='adam', loss='mse'):

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


def get_multiple_output_model_skip_v4(width, depth, activation, input_shape=3,
                                      optimizer='adam', loss='mse'):

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


def get_multiple_output_model_skip_v5(width, depth, activation, input_shape=3,
                                      optimizer='adam', loss='mse'):

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

    leng1 = keras.layers.Dense(1024, activation=activation)(x)
    leng2 = keras.layers.Dense(1024, activation=activation)(leng1)
    leng3 = keras.layers.Dense(1024, activation=activation)(leng2 + leng1)
    leng4 = keras.layers.Dense(1024, activation=activation)(leng3)
    leng5 = keras.layers.Dense(1024, activation=activation)(leng4 + leng1)
    leng6 = keras.layers.Dense(1024, activation=activation)(leng5)
    lengths = keras.layers.Dense(2)(leng6)

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


def get_no_skip_model(width, activation, input_shape=3, output_shape=8,
                      optimizer='adam', loss='mse'):
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


def get_skip_model(width, activation, input_shape=3, output_shape=8,
                   optimizer='adam', loss='mse'):
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


def get_skip_model_v2(width, activation, input_shape=3, output_shape=8,
                      optimizer='adam', loss='mse'):
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
