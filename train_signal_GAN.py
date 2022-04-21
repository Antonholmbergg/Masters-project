import os
#from gpuutils import GpuUtils
#GpuUtils.allocate(gpu_count=1, framework='keras')
#import def_signal_GAN
import cWGANGP_model_def
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=3, suppress=True)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


data = np.load('/mnt/md0/aholmberg/data/signal_had_12_5deg.npy')
condition = data[:, :2]
shower_n = data[:, 3]
signals = data[:, 3:]
signals_filtered = np.load('/mnt/md0/aholmberg/data/signal_had_12_filtered_5deg.npy')


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        logs['batch'] = batch
        self.losses.append(logs)


N = 896
condition_norm = condition.copy()
condition_norm[:, 0] = np.log(condition_norm[:, 0])
cond_scaler = MinMaxScaler().fit(condition_norm)
condition_norm = cond_scaler.transform(condition_norm)

SIGNAL_LENGTH = signals.shape[1]
N_SHOWERS = 10
BATCH_SIZE = 64

latent_dim = 112
generator_in_channels = latent_dim
discriminator_in_channels = N

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9, decay=0
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9, decay=0
)

name = 'critic_conv_gen_trans_adam_5deg_incept1'
epochs = 100
monitor_cond = condition_norm[-40::10,:]

plot_path = f'GAN_plots/{name}/'
if not os.path.isdir(plot_path):
    os.mkdir(plot_path)

history = LossHistory()
monitor = cWGANGP_model_def.GANMonitor(monitor_cond, plot_path, num_signal=4, latent_dim=latent_dim)
cbk = [history, monitor]

g_model = cWGANGP_model_def.get_generator_model_transconv(generator_in_channels)
print(g_model.summary())
d_model = cWGANGP_model_def.get_simpler_critic_incept(discriminator_in_channels)
print(d_model.summary())

wgan = cWGANGP_model_def.WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=10,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=cWGANGP_model_def.generator_loss,
    d_loss_fn=cWGANGP_model_def.discriminator_loss,
)

test_split = 0.1
ind = int(signals_filtered.shape[0]*(1-test_split))

train_signals = signals_filtered[:ind, :]
train_condition = condition_norm[:ind, :]

normalized_signals = np.zeros_like(train_signals)
for i in range(train_signals.shape[0]):
    normalized_signals[i, :] = train_signals[i, :]*(1e19/condition[i, 0])


train_data = [train_condition, normalized_signals]

wgan.fit(train_data, batch_size=BATCH_SIZE, epochs=epochs, callbacks=cbk, shuffle=True)

wgan.generator.save(f'/mnt/md0/aholmberg/GAN_models/gen_{name}')
wgan.discriminator.save(f'/mnt/md0/aholmberg/GAN_models/crit_{name}')

loss_df = pd.DataFrame(history.losses)
loss_df.to_pickle(f'GAN_losses/history_{name}.pkl')
