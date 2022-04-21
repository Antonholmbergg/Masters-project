import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def get_discriminator_model_conv(discriminator_in_channels=896,
                                 condition_shape=2):
    signal_input = layers.Input(shape=(discriminator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x = layers.Conv1D(filters=32, kernel_size=9, padding='same',
                      kernel_initializer='he_normal')(signal_input)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=32, kernel_size=9, padding='same',
                      kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(filters=64, kernel_size=9, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=9, padding='same',
                      kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(filters=128, kernel_size=9, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=9, padding='same',
                      kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=2, kernel_size=9, padding='same',
                      kernel_initializer='he_normal', strides=2)(x)
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
        name="discriminator"
        )
    return d_model


def get_simpler_critic(discriminator_in_channels=896,
                                 condition_shape=2):
    signal_input = layers.Input(shape=(discriminator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    #x = layers.LayerNormalization()(signal_input)
    x = layers.Conv1D(filters=2, kernel_size=15, padding='same',
                      kernel_initializer='he_normal', strides=4)(signal_input)
    x = layers.LeakyReLU()(x)
    
    #x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=8, kernel_size=15, padding='same',
                      kernel_initializer='he_normal', strides=2)(x)
    x = layers.LeakyReLU()(x)


    #x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    condition_input_expanded = layers.Dense(112)(condition_input)
    x = tf.concat((x, condition_input_expanded), axis=1)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    #x = layers.Dropout(0.25)(x)
    x = layers.Dense(1)(x)

    c_model = keras.models.Model(
        [signal_input, condition_input],
        x,
        name="critic"
        )
    return c_model


def get_simpler_critic_incept(discriminator_in_channels=896,
                              condition_shape=2):
    signal_input = layers.Input(shape=(discriminator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))

    x1 = layers.Conv1D(filters=8, kernel_size=15, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x1 = layers.LeakyReLU()(x1)
    x2 = layers.Conv1D(filters=8, kernel_size=25, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x2 = layers.LeakyReLU()(x2)
    x3 = layers.Conv1D(filters=8, kernel_size=5, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x3 = layers.LeakyReLU()(x3)
    x4 = layers.Conv1D(filters=8, kernel_size=35, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x4 = layers.LeakyReLU()(x4)
    x5 = layers.Conv1D(filters=8, kernel_size=45, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x5 = layers.LeakyReLU()(x5)
    x6 = layers.Conv1D(filters=8, kernel_size=55, padding='same',
                       kernel_initializer='he_normal', strides=4)(signal_input)
    x6 = layers.LeakyReLU()(x6)

    x = tf.concat((x1, x2, x3, x4, x5, x6), axis=2)

    x = layers.Conv1D(filters=1, kernel_size=1, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    #x = layers.LayerNormalization()(x)
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
    def __init__(self, out_ch, k, act, strides=1):
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
    noise = layers.Input(shape=(generator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))
    x = layers.Dense(generator_in_channels)(condition_input)
    x = layers.Reshape((generator_in_channels, 1))(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal')(tf.concat((x, noise), axis=-1))
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=8, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=8, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=4, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)

    x = layers.Flatten()(x)
    g_model = keras.models.Model([noise, condition_input], x, name="generator")
    return g_model


def get_generator_model_transconv(generator_in_channels=112, condition_shape=2):
    noise = layers.Input(shape=(generator_in_channels, 1))
    condition_input = layers.Input(shape=(condition_shape))
    x = layers.Dense(generator_in_channels)(condition_input)
    x = layers.Reshape((generator_in_channels, 1))(x)

    x = layers.Conv1D(filters=32, kernel_size=9, padding='same', kernel_initializer='he_normal')(tf.concat((x, noise), axis=-1))
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=16, kernel_size=9, strides=2, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=16, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=8, kernel_size=9, strides=2, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=8, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(filters=4, kernel_size=9, strides=2, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=4, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters=1, kernel_size=9, padding='same', kernel_initializer='he_normal')(x)

    x = layers.Flatten()(x)
    g_model = keras.models.Model([noise, condition_input], x, name="generator")
    return g_model


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_signals, fake_signals, condition):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)  # same dimension as batches
        interpolated = alpha*real_signals + (1 - alpha)*fake_signals
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal
            pred = self.discriminator([interpolated, condition], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated, condition])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, train_data):
        if isinstance(train_data, tuple):
            train_data = train_data[0]
        """
        real_signals = train_data[0]
        condition = train_data[1] """
        condition, real_signals = train_data

        #condition = real_signals[:,:2]
        #real_signals = real_signals[:,2:]
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:

                fake_signals = self.generator([random_latent_vectors, condition], training=True)
                fake_logits = self.discriminator([fake_signals, condition], training=True)
                real_logits = self.discriminator([real_signals, condition], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_sig=real_logits, fake_sig=fake_logits)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_signals, fake_signals, condition)
                
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
            )
        with tf.GradientTape() as tape:

            generated_signals = self.generator(
                [random_latent_vectors, condition], training=True
                )

            gen_signal_logits = self.discriminator(
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
        return {"d_loss": d_loss, "g_loss": g_loss, "d_cost": d_cost, "gp": gp}


class GANMonitor(keras.callbacks.Callback):
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


def discriminator_loss(real_sig, fake_sig):
    real_loss = tf.reduce_mean(real_sig)
    fake_loss = tf.reduce_mean(fake_sig)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_sig):
    return -tf.reduce_mean(fake_sig)


if __name__ == "__main__":
    g_model = get_generator_model_conv()
    print(g_model.summary())
    d_model = get_critic_model_inceptionlike()
    print(d_model.summary())
