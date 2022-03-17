import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def get_discriminator_model(discriminator_in_channels):
    input = layers.Input(shape=discriminator_in_channels)
    x = layers.Dense(512, activation='relu')(input)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(input, x, name="discriminator")
    return d_model


def get_generator_model(generator_in_channels):
    noise = layers.Input(shape=(generator_in_channels,))
    x = layers.Dense(1024, activation='relu')(noise)
    x = layers.Dense(896, activation='relu')(x)
    x = layers.Dense(896)(x)
    g_model = keras.models.Model(noise, x, name="generator")
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
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0) #same dimension as batches
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff
        interpolated_with_condition = tf.concat((condition, interpolated), 1)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_with_condition)
            # 1. Get the discriminator output for this interpolated signal
            pred = self.discriminator(interpolated_with_condition, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated_with_condition])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))# , 2, 3])) again need same nr of dims as the batches
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_signals):
        if isinstance(real_signals, tuple):
            real_signals = real_signals[0]

        condition = real_signals[:,:2]
        real_signals = real_signals[:,2:]
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            generator_input = tf.concat((condition, random_latent_vectors), axis=1)
            discriminator_input_real = tf.concat((condition, real_signals), axis=1)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_signals = self.generator(generator_input, training=True)
                # Get the logits for the fake images
                discriminator_input_fake = tf.concat((condition, fake_signals), 1)
                fake_logits = self.discriminator(discriminator_input_fake, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(discriminator_input_real, training=True)

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
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        latent_vec_condition = tf.concat((condition, random_latent_vectors), 1)
        with tf.GradientTape() as tape:
            # Generate fake signals using the generator
            generated_signals = self.generator(latent_vec_condition, training=True)
            # Get the discriminator logits for fake signals
            gen_sig_cond = tf.concat((condition, generated_signals), 1)
            gen_signal_logits = self.discriminator(gen_sig_cond, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_signal_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, condition, num_signal=4, latent_dim=896):
        self.num_sig = num_signal
        self.latent_dim = latent_dim
        self.condition = condition

    def on_epoch_end(self, epoch, logs=None):
        if epoch%10 == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_sig, self.latent_dim))
            generator_input = tf.concat((self.condition, random_latent_vectors), 1)
            generated_signals = self.model.generator(generator_input)

        
            for i in range(self.num_sig):
                signal = generated_signals[i].numpy()
                plt.ioff()
                fig = plt.figure()
                plt.plot(signal)
                plt.savefig(f'GAN_plots/generated_signal_{i}_{epoch}.png')
                plt.close(fig)


def discriminator_loss(real_sig, fake_sig):
    real_loss = tf.reduce_mean(real_sig)
    fake_loss = tf.reduce_mean(fake_sig)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_sig):
    return -tf.reduce_mean(fake_sig)

