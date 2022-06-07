"""Runs a random grid search to optimise hyper parameters for the signal GAN.
Saves all models in a specified directory.
"""
import os
import cWGANGP_model_def
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from NuRadioReco.utilities import units
import random


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

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# to keep track of loss while training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        logs['batch'] = batch
        self.losses.append(logs)


def main(g_model, c_model, signals_filtered, condition, model_name, model_dir='/mnt/md0/aholmberg/GAN_models/',
         lr_g=0.0001, lr_c=0.0001, epochs=100, latent_dim=112, BATCH_SIZE=64, e_scaling=True, critic_extra_steps=10):
    """Trains one instance of the GAN and saves it

    Args:
        g_model (keras model): the generator model
        c_model (keras model): the critic model
        signals_filtered (numpy array): the filtered training signals
        condition (numpy arry): the training conditions belonging to the filtered signals
        model_name (str): the name to save the model as
        model_dir (str, optional): a directory do save the models in. Defaults to '/mnt/md0/aholmberg/GAN_models/'.
        lr_g (float, optional): learning rate for the generators optimiser . Defaults to 0.0001.
        lr_c (float, optional): learning rate for the critics optimiser. Defaults to 0.0001.
        epochs (int, optional): number of epochs to train the model. Defaults to 100.
        latent_dim (int, optional): size of the latent space. Defaults to 112.
        BATCH_SIZE (int, optional): size of the minibatches. Defaults to 64.
        e_scaling (bool, optional): to use scaling only dependent on energy (True) or one that depends on both energy and angle (False). Defaults to True.
        critic_extra_steps (int, optional): number of times the critic is trained for every time the generator is trained. Defaults to 10.
    """

    n_index = 1.78
    cherenkov_angle = np.arccos(1. / n_index)

    # scale the condition to range (0,1)
    condition_norm = condition.copy()
    condition_norm[:, 0] = (np.log10(condition_norm[:, 0]) - 15)/(19 - 15)
    
    # Change this depending on the range of angles used
    #condition_norm[:, 1] = ((condition_norm[:, 1] - cherenkov_angle) / units.deg + 2.5) / 5    # +- 2.5deg
    condition_norm[:, 1] = ((condition_norm[:, 1] - cherenkov_angle) / units.deg + 5) / 10  #  +- 5deg

    # use 50% of the data to train
    test_split = 0.5
    ind = int(signals_filtered.shape[0]*test_split)

    train_signals = signals_filtered[:ind, :]
    train_condition = condition_norm[:ind, :]
    scaling_condition = condition[:ind, :]

    # scale the signals depending on only energy or both energy and angle
    if e_scaling:
        normalized_signals = np.zeros_like(train_signals)
        for i in range(train_signals.shape[0]):
            normalized_signals[i, :] = train_signals[i, :]*(1e19/condition[i, 0])
    else:
        energy_scale = np.expand_dims(1e19/scaling_condition[:, 0], axis=-1)
        #angle_scale = np.expand_dims((((scaling_condition[:, 1]/units.deg - cherenkov_angle/units.deg))**4 + 1)/3, axis=-1)  #+-2.5deg
        angle_scale = np.expand_dims((((scaling_condition[:, 1]/units.deg - cherenkov_angle/units.deg))**4 + 1)/6, axis=-1)  #+-5deg
        normalized_signals = train_signals * energy_scale * angle_scale

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=lr_g, beta_1=0.0, beta_2=0.9, decay=0
    )
    critic_optimizer = keras.optimizers.Adam(
        learning_rate=lr_c, beta_1=0.0, beta_2=0.9, decay=0
    )

    # set up monitor to plot the four last signals i nthe dataset every 10 epochs
    plot_path = f'GAN_plots/{model_name}/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    monitor_cond = condition_norm[-40::10, :]
    monitor = cWGANGP_model_def.GANMonitor(monitor_cond, plot_path, num_signal=4, latent_dim=latent_dim)
    # to keep track of the loss, there is probably a better way of doing this but it works
    history = LossHistory()
    cbk = [history, monitor]

    wgan = cWGANGP_model_def.WGAN(
        critic=c_model,
        generator=g_model,
        latent_dim=latent_dim,
        critic_extra_steps=critic_extra_steps,
    )

    wgan.compile(c_optimizer=critic_optimizer, g_optimizer=generator_optimizer,
                 g_loss_fn=cWGANGP_model_def.generator_loss, c_loss_fn=cWGANGP_model_def.critic_loss)

    train_data = [train_condition, normalized_signals]

    wgan.fit(train_data, batch_size=BATCH_SIZE, epochs=epochs, callbacks=cbk, shuffle=True)

    # Save the generator, critic and loss history
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    wgan.generator.save(f'{model_dir}gen_{model_name}')
    wgan.critic.save(f'{model_dir}crit_{model_name}')

    loss_df = pd.DataFrame(history.losses)
    loss_df.to_pickle(f'GAN_losses/history_{model_name}.pkl')


if __name__ == '__main__':
    data = np.load('/mnt/md0/aholmberg/data/signal_had_14_10deg.npy')
    # the rest of 'data' is unfiltered signal -should fix so not loaded unnecessarily
    condition = data[:, :2]
    del data
    signals_filtered = np.load('/mnt/md0/aholmberg/data/signal_had_14_filtered_10deg.npy')

    # Parameters to search
    params = {'lr': [2e-4, 1e-4, 5e-5],
              'critic_filters': [8, 12, 16, 24],
              'generator_filters': [24, 32, 40, 48],
              'generator_k_size': [5, 7, 9, 11, 13, 15]
              }
    for i in range(0, 50):
        # With `clear_session()` called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        tf.keras.backend.clear_session()
        # Make a random choice of parameters should be changed so duplicate choices are impossible
        config = {}
        for key in params:
            config[key] = random.choice(params[key])
        print(config)

        # Define the generator and critic
        g_model = cWGANGP_model_def.get_generator_model_transconv(
            n_filters=config['generator_filters'], kernel_size=config['generator_k_size'])
        print(g_model.summary())
        c_model = cWGANGP_model_def.get_simpler_critic_incept(n_filters=config['critic_filters'])
        print(c_model.summary())

        # Set the name of the models
        name = ''
        for key in config:
            name += f'{key}={config[key]}-'
        model_name = f'run{i}-{name[:-1]}'
        model_dir = '/mnt/md0/aholmberg/GAN_models/transconv-incept-m14-10deg-05split-fixed/'
        # train and save the model
        main(g_model, c_model, signals_filtered, condition, model_name, lr_c=config['lr'], lr_g=config['lr'],
             model_dir=model_dir, e_scaling=False)
