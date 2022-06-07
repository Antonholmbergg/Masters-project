"""Defines the cWGAN-GP architechture and the training step. 
Also defines different generators and critics. Tested with tensorflow version 2.4
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def get_critic_model_conv(critic_in_channels=896, condition_shape=2):
    """Defines a normal convolutional model to use as critic.
    Does not work particularly well, might be too many filters.

    Args:
        critic_in_channels (int, optional): in case the signal shape is defined differently. Defaults to 896.
        condition_shape (int, optional): in case the condition shape is defined differently. Defaults to 2.

    Returns:
        _type_: An uncomipled model with inputs of condition and signal and output of size 1
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    # Each bloch halvs the dimension
    x = layers.Conv1D(filters=32, kernel_size=9, padding='same', kernel_initializer='he_normal')(signal_input)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=32, kernel_size=9, padding='same', kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(filters=64, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=9, padding='same', kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(filters=128, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=9, padding='same', kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(filters=2, kernel_size=9, padding='same', kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return d_model


def get_simpler_critic(critic_in_channels=896, condition_shape=2, kernel_size=25, n_filters=2):
    """Simpler straight convolutional critic. Works OK.

    Args:
        critic_in_channels (int, optional): _description_. Defaults to 896.
        condition_shape (int, optional): _description_. Defaults to 2.
        kernel_size (int, optional): the kernel size in all of the convolutional layers. Defaults to 25.
        n_filters (int, optional): number of filters in the first layer, then scaled opposit to how the dimension scales. Defaults to 2.

    Returns:
        _type_: uncompiled keras model
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same',
                      kernel_initializer='he_normal')(signal_input)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(filters=n_filters*4, kernel_size=kernel_size, padding='same',
                      kernel_initializer='he_normal', strides=4)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(filters=n_filters*16, kernel_size=kernel_size, padding='same',
                      kernel_initializer='he_normal', strides=4)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(filters=1, kernel_size=1, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1)(x)

    c_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return c_model


def get_simpler_critic_incept(critic_in_channels=896, condition_shape=2, n_filters=8):
    """Simpler version of the inception-like critic with no layer normalization. 
    Is the one which reached the best perforance so far.

    Args:
        critic_in_channels (int, optional): size of the signal. Defaults to 896.
        condition_shape (int, optional): size of the condition. Defaults to 2.
        n_filters (int, optional): number of filters in the incept-block in the model. Defaults to 8.

    Returns:
        _type_: uncompiled keras model
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x1 = layers.Conv1D(filters=n_filters, kernel_size=15, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x1 = layers.LeakyReLU()(x1)
    x2 = layers.Conv1D(filters=n_filters, kernel_size=25, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x2 = layers.LeakyReLU()(x2)
    x3 = layers.Conv1D(filters=n_filters, kernel_size=5, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x3 = layers.LeakyReLU()(x3)
    x4 = layers.Conv1D(filters=n_filters, kernel_size=35, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x4 = layers.LeakyReLU()(x4)
    x5 = layers.Conv1D(filters=n_filters, kernel_size=45, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x5 = layers.LeakyReLU()(x5)
    x6 = layers.Conv1D(filters=n_filters, kernel_size=55, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x6 = layers.LeakyReLU()(x6)

    x = tf.concat((x1, x2, x3, x4, x5, x6), axis=2)

    x = layers.Conv1D(filters=1, kernel_size=1, padding='same',
                      kernel_initializer='he_normal')(x)

    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1)(x)

    c_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return c_model


class conv_ln_LReLU_block(layers.Layer):
    """Defines a block of 1-d convolution -> layer normalisation -> leakyReLU.
    This is not used much since layer norm did not seem to work very well.
    """
    def __init__(self, out_ch, k, act, strides=1):
        """initializes the parameters

        Args:
            out_ch (int): number of channels
            k (int): kernel size
            act (_type_): activation (should be Leakyrelu)
            strides (int, optional): what striding to use, if any. Defaults to 1.
        """
        super().__init__()
        self.out_ch = out_ch
        self.k = k
        self.act = act
        self.strides = strides

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.ln(x)
        x = self.act(x)
        return x

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            self.out_ch, self.k, padding='same',
            kernel_initializer='he_uniform', strides=self.strides
            )
        self.ln = layers.LayerNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_ch': self.out_ch,
            'k': self.k,
            'act': self.act,
            'strides': self.strides,
        })
        return config


class ConcatBlockConv5(layers.Layer):
    """Uses the conv_ln_LReLU_block to define an inception-like block,
    not used much because layer norm seems to not work very well
    """
    def __init__(self, out_ch, k, act=layers.LeakyReLU()):
        super().__init__()
        self.out_ch = out_ch
        self.k = k
        self.act = act

    def call(self, x):
        x = tf.concat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), self.c5(x), self.pool(x)], axis=2)
        x = self.c6(x)
        return x

    def build(self, input_shape):
        self.pool = layers.MaxPool1D(pool_size=2, strides=2, padding='same')
        self.c1 = conv_ln_LReLU_block(self.out_ch, self.k, self.act, strides=2)
        self.c2 = conv_ln_LReLU_block(self.out_ch, self.k * 2, self.act, strides=2)
        self.c3 = conv_ln_LReLU_block(self.out_ch, self.k // 2, self.act, strides=2)
        self.c4 = conv_ln_LReLU_block(self.out_ch, self.k // 4, self.act, strides=2)
        self.c5 = conv_ln_LReLU_block(self.out_ch, self.k * 4, self.act, strides=2)
        self.c6 = conv_ln_LReLU_block(self.out_ch, 1, self.act, strides=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_ch': self.out_ch,
            'k': self.k,
            'act': self.act
        })
        return config


def get_critic_model_inceptionlike(critic_in_channels=896, condition_shape=2):
    """Defines an inception like model with multiple blocks, does not work very well.
    The model is probably too complex and I'm not sure if layering those blocks even makes sense.

    Args:
        critic_in_channels (int, optional): size of the signal. Defaults to 896.
        condition_shape (int, optional): size of the condition. Defaults to 2.

    Returns:
        _type_: uncompiled keras model
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x = ConcatBlockConv5(32, 32)(signal_input)
    x = ConcatBlockConv5(64, 32)(x)
    x = ConcatBlockConv5(1, 32)(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic_inceptionlike"
        )
    return d_model


def get_generator_model_conv(generator_in_channels=112, condition_shape=2):
    """Generator model which uses upsampling instead of transpose convolutions. Works worse.

    Args:
        generator_in_channels (int, optional): size of the signal. Defaults to 112.
        condition_shape (int, optional): siza of the condition. Defaults to 2.

    Returns:
        _type_: uncompiled keras model
    """
    kwargs = {'kernel_size': 3, 'padding': 'same', 'kernel_initializer': 'he_normal'}

    noise = layers.Input(shape=(generator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))
    x = layers.Dense(generator_in_channels)(condition_input)
    x = layers.Reshape((generator_in_channels, 1))(x)

    x = layers.Conv1D(filters=32, **kwargs)(tf.concat((x, noise), axis=-1))
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=32, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=16, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=16, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=8, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=8, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=4, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=1, **kwargs)(x)

    x = layers.Flatten()(x)
    g_model = keras.models.Model([noise, condition_input], x, name="generator")
    return g_model


def get_simpler_critic_incept_v2(critic_in_channels=896, condition_shape=2):
    """Same start as the original version but followed by some conv layers instead of dense layers immediatly.
    At the moment it does not work well but deserves more investigation.

    Args:
        critic_in_channels (int, optional): size of the signal. Defaults to 896.
        condition_shape (int, optional): size of the condition. Defaults to 2.

    Returns:
        _type_: uncompiiled keras model
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x1 = layers.Conv1D(filters=16, kernel_size=15, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x1 = layers.LeakyReLU()(x1)
    x2 = layers.Conv1D(filters=16, kernel_size=25, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x2 = layers.LeakyReLU()(x2)
    x3 = layers.Conv1D(filters=16, kernel_size=5, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x3 = layers.LeakyReLU()(x3)
    x4 = layers.Conv1D(filters=16, kernel_size=35, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x4 = layers.LeakyReLU()(x4)
    x5 = layers.Conv1D(filters=16, kernel_size=45, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x5 = layers.LeakyReLU()(x5)
    x6 = layers.Conv1D(filters=16, kernel_size=55, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x6 = layers.LeakyReLU()(x6)

    x = tf.concat((x1, x2, x3, x4, x5, x6), axis=2)

    x = layers.Conv1D(filters=16, kernel_size=25, padding='same',
                      kernel_initializer='he_normal', strides=4)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(filters=32, kernel_size=25, padding='same',
                      kernel_initializer='he_normal', strides=4)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1)(x)

    c_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return c_model


def get_simpler_critic_incept_v3(critic_in_channels=896, condition_shape=2):
    """Same start as the original but folowed by 2-d conv layers.
    Does not perform very well at the moment but could be worth investigating.

    Args:
        critic_in_channels (int, optional): size of the signal. Defaults to 896.
        condition_shape (int, optional): size of the condition. Defaults to 2.

    Returns:
        _type_: uncompiled keras model
    """
    signal_input = layers.Input(shape=(critic_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x1 = layers.Conv1D(filters=16, kernel_size=15, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x1 = layers.LeakyReLU()(x1)
    x2 = layers.Conv1D(filters=16, kernel_size=25, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x2 = layers.LeakyReLU()(x2)
    x3 = layers.Conv1D(filters=16, kernel_size=5, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x3 = layers.LeakyReLU()(x3)
    x4 = layers.Conv1D(filters=16, kernel_size=35, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x4 = layers.LeakyReLU()(x4)
    x5 = layers.Conv1D(filters=16, kernel_size=45, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x5 = layers.LeakyReLU()(x5)
    x6 = layers.Conv1D(filters=16, kernel_size=55, padding='same',
                       kernel_initializer='he_normal', strides=2)(signal_input)
    x6 = layers.LeakyReLU()(x6)

    x = tf.concat((x1, x2, x3, x4, x5, x6), axis=2)

    x = layers.Conv1D(filters=56, kernel_size=25, padding='same',
                      kernel_initializer='he_normal', strides=8)(x)
    x = layers.LeakyReLU()(x)

    x = tf.expand_dims(x, axis=-1)

    condition_input_expanded = layers.Dense(56*56)(condition_input)
    condition_input_filter = layers.Reshape((56, 56, 1))(condition_input_expanded)
    x = tf.concat((x, condition_input_filter), axis=-1)

    x = layers.Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2), kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1)(x)

    c_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return c_model


def get_generator_model_transconv(generator_in_channels=112, condition_shape=2, n_filters=32, kernel_size=9):
    """A generator inspired by DCGAN, but with extra conv layers and without batchnormalization.
    Works fairly well.

    Args:
        generator_in_channels (int, optional): size of the latent space. Defaults to 112.
        condition_shape (int, optional): size of the condition parameter. Defaults to 2.
        n_filters (int, optional): number of filters in the first layer, this is halved every time the dimension is doubled. Defaults to 32.
        kernel_size (int, optional): kernel size for all layers in the generator. Defaults to 9.

    Returns:
        _type_: uncompiled keras model
    """
    kwargs = {'kernel_size': kernel_size, 'padding': 'same', 'kernel_initializer': 'he_normal'}
    noise = layers.Input(shape=(generator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))
    x = layers.Dense(generator_in_channels)(condition_input)
    x = layers.Reshape((generator_in_channels, 1))(x)

    x = layers.Conv1D(filters=n_filters, **kwargs)(tf.concat((x, noise), axis=-1))
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=n_filters/2, strides=2, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=n_filters/2, **kwargs)(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=n_filters/4, strides=2, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=n_filters/4, **kwargs)(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=n_filters/8, strides=2, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=n_filters/8, **kwargs)(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=1, **kwargs)(x)

    x = layers.Flatten()(x)
    g_model = keras.models.Model([noise, condition_input], x, name="generator")
    return g_model


def get_generator_simpler_transconv(generator_in_channels=112, condition_shape=2, n_filters=16, kernel_size=25):
    """Same as the standard but without the extra conv layers between the transpose layers

    Args:
        generator_in_channels (int, optional): size of the latent space. Defaults to 112.
        condition_shape (int, optional): size of the condition. Defaults to 2.
        n_filters (int, optional): number of filters in the first layer. Defaults to 16.
        kernel_size (int, optional): kernel size in all layers. Defaults to 25.

    Returns:
        _type_: uncompiled keras model
    """
    kwargs = {'kernel_size': kernel_size, 'strides': 2, 'padding': 'same', 'kernel_initializer': 'he_normal'}
    noise = layers.Input(shape=(generator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))
    x = layers.Dense(generator_in_channels)(condition_input)
    x = layers.Reshape((generator_in_channels, 1))(x)

    x = layers.Conv1DTranspose(filters=n_filters, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1DTranspose(filters=n_filters/2, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1DTranspose(filters=n_filters/4, **kwargs)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=n_filters/4, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    x = layers.Flatten()(x)
    g_model = keras.models.Model([noise, condition_input], x, name="generator")
    return g_model


class WGAN(keras.Model):
    """Defines the cWGAN-GP and the training step"""
    def __init__(self, critic, generator, latent_dim, critic_extra_steps=5, gp_weight=10.0):
        """Initiates the WGAN

        Args:
            critic (keras model): the model to use as a critic, needs to take a signal and a condition input and have an output of size 1
            generator (keras model): The model to use as the generator, needs to take a latent input as well as a condition input, output is a signal that matches the size of the critic.
            latent_dim (int): size of the latent space
            critic_extra_steps (int, optional): how many times the critic is trained for every time the generator is trained. Defaults to 5.
            gp_weight (float, optional): how the gradient penalty is weighted in the loss function. Defaults to 10.0.
        """
        super(WGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.c_steps = critic_extra_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        """Compiles the model

        Args:
            c_optimizer (keras optimizer): the optimiser for the critic
            g_optimizer (keras optimizer): the optimizer for the generator
            c_loss_fn (function): The loss function for the critic
            g_loss_fn (function): The loss function for the generator
        """
        super(WGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_signals, fake_signals, condition):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated signal
        and added to the critic loss.
        """
        # Get the interpolated signal
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)  # same dimension as batches
        interpolated = alpha*real_signals + (1 - alpha)*fake_signals
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated signal
            pred = self.critic([interpolated, condition], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated, condition])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # @tf.function <---- does not work unless tensorflow is updated or python downgraded should give speedup
    def train_step(self, train_data):
        """The training step used when using .fit() on keras model

        Args:
            train_data (list): The training data in the shape: (condition, real signals)

        Returns:
            _type_: dictionary of the losses
        """
        if isinstance(train_data, tuple): #  somtimes the data is packaged as touple
            train_data = train_data[0]

        condition, real_signals = train_data

        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # This is actually implemented incorectly (just realised).
        # It trains on the same real data and conditions ten times 
        # instead of sampling new data every iteration like you should.
        # So there is a risk that it can overfit in a strange way.
        # Sould be fixed to see if it affects performance
        for i in range(self.c_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:

                fake_signals = self.generator([random_latent_vectors, condition], training=True)
                fake_logits = self.critic([fake_signals, condition], training=True)
                real_logits = self.critic([real_signals, condition], training=True)

                # Calculate the critic loss using the fake and real image logits
                d_cost = self.c_loss_fn(real_sig=real_logits, fake_sig=fake_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_signals, fake_signals, condition)

                # Add the gradient penalty to the original critic loss
                c_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the critic loss
            d_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )

        # Train the generator
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
            )
        with tf.GradientTape() as tape:

            generated_signals = self.generator(
                [random_latent_vectors, condition], training=True
                )

            gen_signal_logits = self.critic(
                [generated_signals, condition], training=True
                )

            g_loss = self.g_loss_fn(gen_signal_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(
            g_loss, self.generator.trainable_variables
            )
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"c_loss": c_loss, "g_loss": g_loss, "c_cost": d_cost, "gp": gp}


class GANMonitor(keras.callbacks.Callback):
    """Plots a few example signals every ten epochs"""
    def __init__(self, condition, path, num_signal=4, latent_dim=896):
        self.num_sig = num_signal
        self.latent_dim = latent_dim
        self.condition = condition
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            random_latent_vectors = tf.random.normal(
                shape=(self.num_sig, self.latent_dim)
                )
            generated_signals = self.model.generator(
                [random_latent_vectors, self.condition]
                )

            for i in range(self.num_sig):
                signal = generated_signals[i].numpy()
                plt.ioff()
                fig = plt.figure()
                plt.plot(signal)
                plt.savefig(self.path + f'generated_signal_{i}_{epoch}.png')
                plt.close(fig)

# Define the loss function for the critic
def critic_loss(real_sig, fake_sig):
    real_loss = tf.reduce_mean(real_sig)
    fake_loss = tf.reduce_mean(fake_sig)
    return fake_loss - real_loss


# Define the loss function for the generator
def generator_loss(fake_sig):
    return -tf.reduce_mean(fake_sig)


if __name__ == "__main__":
    # just for testing that models compile as expected without having to start a trainning run
    g_model = get_generator_simpler_transconv()
    print(g_model.summary())
    d_model = get_simpler_critic()
    print(d_model.summary())
