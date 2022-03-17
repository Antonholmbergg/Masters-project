import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn.preprocessing import MinMaxScaler
import raytrace_model_def
from pickle import dump

tf.random.set_seed(
    42
)


#df = pd.read_csv('/mnt/md0/aholmberg/data/raytrace_samples_sobol_21.csv')
df = pd.read_csv('/mnt/md0/aholmberg/data/raytrace_samples_sobol_24.csv')

sc_pos_r = df['source_pos_r'].to_numpy().astype(np.float64)
sc_pos_z = df['source_pos_z'].to_numpy().astype(np.float64)
ant_pos_z = df['antenna_pos_z'].to_numpy().astype(np.float64)
x = np.stack((sc_pos_r, sc_pos_z, ant_pos_z), axis=1)

travel_time = df['travel_time'].to_numpy().astype(np.float64)
path_length = df['path_length'].to_numpy().astype(np.float64)
launch = df['launch_angle'].to_numpy().astype(np.float64)
recieve = df['recieve_angle'].to_numpy().astype(np.float64)
y = np.stack((travel_time, path_length, launch, recieve), axis=1)

unique, index, count = np.unique(x, return_counts=True, return_index=True, axis=0)
print(unique, index, count)
print(np.unique(count, return_counts=True))
x[index[count == 1], :]
x_new = np.delete(x, index[count == 1], axis=0)
y_new = np.delete(y, index[count == 1], axis=0)
unique, index, count = np.unique(x_new, return_counts=True, return_index=True, axis=0)
print(unique, index, count)
print(np.unique(count, return_counts=True))


x_train = x_new[0::2,:]

y_temp1  = y_new[0::2,:]
y_temp2  = y_new[1::2,:]
y_new = np.zeros((y_temp1.shape[0], 8))



for i in range(4):
    y_new[:,2*i] = y_temp1[:,i]
    y_new[:,2*i+1] = y_temp2[:,i]

y_train = y_new


scaler_x = MinMaxScaler(feature_range=(0,1))
scaler_x.fit(x_train)
norm_x_train = scaler_x.transform(x_train)

scaler_y = MinMaxScaler(feature_range=(0,1))
scaler_y.fit(y_train)
norm_y_train = scaler_y.transform(y_train)

path = '/mnt/md0/aholmberg/models/multi-out-bigger-skip-weight-v2-2'

scaler_path_x = path + "-x-scaler.pkl"
scaler_path_y = path + "-y-scaler.pkl"

dump(scaler_x, open(scaler_path_x, 'wb'))
dump(scaler_y, open(scaler_path_y, 'wb'))

#single output model
"""
activation = 'relu'
depth = 6
width = 1024
opt = keras.optimizers.Adam()
model = raytrace_model_def.get_simple_model(width, depth, activation, output_shape=2)
#model = raytrace_model_def.get_simple_model_dropout(width, depth, activation, optimizer=opt, batch_norm=True)
#model = raytrace_model_def.get_skip_model_2(width, activation)
print(model.summary())
"""


#"""
#multi output model
activation = 'relu'
depth = 1
width = 1024
opt = keras.optimizers.Adam(learning_rate=1e-3)
model = raytrace_model_def.get_multiple_output_model_skip_v2_2(width, depth, activation=activation)
#"""

def scheduler(epoch, lr):
    if epoch <= 1:
        return 1e-2
    elif epoch == 3:
        return 2e-3
    else:
       return lr * tf.math.exp(-0.035)

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = '/mnt/md0/aholmberg/models/multi-out-bigger-skip-weight-v2-2'

if not os.path.isdir(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    save_weights_only=True
)

#single output model
"""
model.fit(
    norm_x_train,
    norm_y_train[:,:2],
    epochs=100,
    batch_size=512,
    validation_split=0.1,
    callbacks=[lr_scheduler, model_checkpoint_callback]
    )
"""

# multi output model
"""
model.fit(
    norm_x_train,
    [norm_y_train[:,:2], norm_y_train[:,2:4], norm_y_train[:,4:]],
    epochs=150,
    batch_size=128,
    validation_split=0.1,
    callbacks=[lr_scheduler, model_checkpoint_callback]
    )
"""
#"""
model.fit(
    norm_x_train,
    [norm_y_train[:,:2], norm_y_train[:,2:4], norm_y_train[:,4:6], norm_y_train[:,6:]],
    epochs=200,
    batch_size=512,
    validation_split=0.1,
    callbacks=[lr_scheduler, model_checkpoint_callback]
    )

model.load_weights(checkpoint_filepath)
#"""


if not os.path.isdir(path):
    os.mkdir(path)

model.save(path)

