import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
#import def_signal_GAN
import cWGANGP_model_def

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

from tensorflow import keras
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn.preprocessing import MinMaxScaler
from NuRadioReco.utilities import units, fft

data = np.load('/mnt/md0/aholmberg/data/signal_had_15.npy')
condition = data[:,:2]
shower_n = data[:,3]
signals = data[:,3:]
signals_filtered = np.load('/mnt/md0/aholmberg/data/signal_had_15_filtered.npy')

N = 896
condition_norm = condition.copy()
condition_norm[:,0] = np.log(condition_norm[:,0])
cond_scaler = MinMaxScaler().fit(condition_norm)
condition_norm  = cond_scaler.transform(condition_norm)

SIGNAL_LENGTH = signals.shape[1]
N_SHOWERS = 10
BATCH_SIZE = 128

latent_dim = 1024
#cond_dim = condition.shape[1]
generator_in_channels = latent_dim# + cond_dim
discriminator_in_channels = N# + cond_dim

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)


epochs = 1000
monitor_cond = condition_norm[-40::10,:]

plot_path = 'GAN_plots/normalized_discriminator_lstm/'
if not os.path.isdir(plot_path):
    os.mkdir(plot_path)

cbk = cWGANGP_model_def.GANMonitor(monitor_cond, plot_path, num_signal=4, latent_dim=latent_dim)
monitor_cond

g_model = cWGANGP_model_def.get_generator_model(generator_in_channels)
print(g_model.summary())
d_model = cWGANGP_model_def.get_discriminator_model_lstm(discriminator_in_channels)
print(d_model.summary())

wgan = cWGANGP_model_def.WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=5,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=cWGANGP_model_def.generator_loss,
    d_loss_fn=cWGANGP_model_def.discriminator_loss,
)

train_signals = signals_filtered[0:-40,:]
train_condition = condition_norm[0:-40,:]

normalized_signals = np.zeros_like(train_signals)
for i in range(train_signals.shape[0]):
    normalized_signals[i,:] = train_signals[i,:]*(1e19/condition[i,0])

train_data = np.concatenate((train_condition, normalized_signals), axis=1)

wgan.fit(train_data, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
wgan.generator.save('/mnt/md0/aholmberg/GAN_models/GAN_generator_normalized_lstm')
wgan.discriminator.save('/mnt/md0/aholmberg/GAN_models/GAN_discriminator_normalized_lstm')