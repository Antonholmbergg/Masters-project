"""Tests the models saved in a directory. For examlpe, from signal_GAN_train.
Saves the results in a dataframe as a .pkl file.
Should be generalized better, currently need to redefine scaling in multiple places in the code -prone to errors.
"""
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy import integrate as quad
from tensorflow import keras
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from NuRadioReco.utilities import units
import pandas as pd

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

# load data
data = np.load('/mnt/md0/aholmberg/data/signal_had_14_10deg.npy')
condition = data[:,:2]
del data
signals_filtered = np.load('/mnt/md0/aholmberg/data/signal_had_14_filtered_10deg.npy')

latent_dim = 112
N = 896
n_index = 1.78
cherenkov_angle = np.arccos(1. / n_index)

# Normalize the conditions, depending on the range of angles used
condition_norm = condition.copy()  # normalize to get range (0,1)
condition_norm[:, 0] = (np.log10(condition_norm[:, 0]) - 15)/(19 - 15)
# 5 deg
# condition_norm[:, 1] = ((condition_norm[:, 1] - cherenkov_angle) / units.deg + 2.5)/ 5
# 10 deg
condition_norm[:, 1] = ((condition_norm[:, 1] - cherenkov_angle) / units.deg + 5)/ 10

# save 50% of data for testing
test_split = 0.5
ind = int(signals_filtered.shape[0]*test_split)

i = 0  # because of how the data is split -means all ten shower profiles for the different conditions
test_signals = signals_filtered[ind+(i%10):, :]
test_conditions = condition[ind+(i%10):, :]
test_conditions_norm = condition_norm[ind+(i%10):, :]
latent_vec = tf.random.normal((test_conditions.shape[0], latent_dim))
# 5 degrees
# normalized_signals = test_signals * (np.expand_dims(1e19/test_conditions[:, 0], axis=-1) * (np.expand_dims(((test_conditions[:, 1]/units.deg - cherenkov_angle/units.deg))**4, axis=-1) + 1)/3)    
# 10 degrees
normalized_signals = test_signals * (np.expand_dims(1e19/test_conditions[:, 0], axis=-1) * (np.expand_dims(((test_conditions[:, 1]/units.deg - cherenkov_angle/units.deg))**4, axis=-1) + 1)/6)    

# specify the name of the directory the models are saved in
run_name = 'transconv-incept-m14-10deg-05split-fixed'
directory = f'/mnt/md0/aholmberg/GAN_models/{run_name}/'
generators = []
critics = []
dirs = []
# find all models in the directory (critics and generators)
for dir in sorted(os.listdir(directory)):
    dirs.append(dir)
generators = dirs[len(dirs)//2:]
critics = dirs[:len(dirs)//2]

# Sort the names of the runs
run_numbers = []
for s in generators:
    temp1 = s[7]
    if s[8].isdigit():
        run_number = int(s[7] + s[8])
    else:
        run_number = int(s[7])
    run_numbers.append(run_number)

run_number_sorted, generators, critics = (list(t) for t in zip(*sorted(zip(run_numbers, generators, critics))))

# test the models
dicts = []
for c, g in zip(critics, generators):
    d = {}
    print(c, g)
    # clear the session to not run out of memory
    tf.keras.backend.clear_session()
    g_model = keras.models.load_model(f'/mnt/md0/aholmberg/GAN_models/{run_name}/{g}/', compile=False)
    g_model.compile()
    pred_signals = g_model.predict([latent_vec, test_conditions_norm])

    # Scaling depends on th range of angles used
    # 5 degrees
    # pred_signals_scaled = pred_signals.copy() / (np.expand_dims(1e19/test_conditions[:, 0], axis=-1) * (np.expand_dims(((test_conditions[:, 1]/units.deg - cherenkov_angle/units.deg))**4, axis=-1) + 1)/3)
    # 10deg 
    pred_signals_scaled = pred_signals.copy() / (np.expand_dims(1e19/test_conditions[:, 0], axis=-1) * (np.expand_dims(((test_conditions[:, 1]/units.deg - cherenkov_angle/units.deg))**4, axis=-1) + 1)/6)

    c_model = keras.models.load_model(f'/mnt/md0/aholmberg/GAN_models/{run_name}/{c}/', compile=False)
    c_model.compile()

    fake_logits = c_model.predict([pred_signals, test_conditions_norm])
    real_logits = c_model.predict([normalized_signals, test_conditions_norm])

    w_dist_approx = np.mean(np.abs(fake_logits-real_logits))  # estimated wasserstein distance

    # Compute the fluence 
    real_energy = quad.simpson(np.power(test_signals, 2), dx=1, axis=-1)
    gen_energy = quad.simpson(np.power(pred_signals_scaled, 2), dx=1, axis=-1)

    # Compute the error in fluence and peak to peak amplitude
    avg_real_energy = np.zeros((real_energy.shape[0]//10, ))
    std_real_energy = np.zeros((real_energy.shape[0]//10, ))
    avg_peak2peak = np.zeros((real_energy.shape[0]//10, ))
    std_peak2peak = np.zeros((real_energy.shape[0]//10, ))
    for i in range(avg_real_energy.shape[0]):
        avg_real_energy[i] = np.mean(real_energy[i*10:i*10+10], axis=0)
        std_real_energy[i] = np.std(real_energy[i*10:i*10+10], axis=0)
        max = np.max(test_signals[i*10:i*10+10], axis=-1)
        min = np.min(test_signals[i*10:i*10+10], axis=-1)
        avg_peak2peak[i] = np.mean(max-min)
        std_peak2peak[i] = np.std(max-min)

    energy_err = np.zeros((pred_signals_scaled.shape[0], ))
    peak_err = np.zeros((pred_signals_scaled.shape[0], ))
    for i in range(pred_signals_scaled.shape[0]):
        err = np.abs(gen_energy[i] - avg_real_energy[i//10])
        if err > std_real_energy[i//10]:
            err = np.sqrt((err - std_real_energy[i//10])**2)/avg_real_energy[i//10]
        else:
            err = 0
        energy_err[i] = err
        
        max = np.max(pred_signals_scaled[i])
        min = np.min(pred_signals_scaled[i])
        p_err = np.abs(avg_peak2peak[i//10] - (max - min))
        if err > std_peak2peak[i//10]:
            p_err = np.sqrt((p_err - std_peak2peak[i//10])**2)/avg_peak2peak[i//10]
        else:
            p_err = 0
        peak_err[i] = p_err

    # Save the found errors and the model name to a dictionary
    peak_err = np.mean(peak_err)
    energy_err = np.mean(energy_err)
    d['peak_err'] = peak_err
    d['energy_err'] = energy_err
    d['w_dist'] = w_dist_approx
    d['name'] = g[4:]
    
    pairs = str(g[4:]).split('-')
    temp = {}
    for pair in pairs[1:]:
        key_val = pair.split('=')

        if len(key_val) == 2:
            temp[f'{key_val[0]}'] = key_val[1]
        else:
            t = temp['lr']
            temp['lr'] = t + '-' + key_val[0]
    d.update(temp)
    dicts.append(d)
    print(peak_err, energy_err, w_dist_approx)

# Save the errors and model names to a dataframe as a .pkl file
data = pd.DataFrame(dicts)
data.to_pickle(f'GAN_losses/signal_gan_results_{run_name}.pkl')
print(data)