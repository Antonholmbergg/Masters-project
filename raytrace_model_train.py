import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

import pandas as pd
from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn.preprocessing import MinMaxScaler
from raytrace_model_def import model_builder
import keras_tuner as kt

df = pd.read_csv('/mnt/md0/aholmberg/data/raytrace_samples_angle.csv')

sc_pos_r = df['source_pos_r'].to_numpy().astype(np.float32)
sc_pos_z = df['source_pos_z'].to_numpy().astype(np.float32)
ant_pos_z = df['antenna_pos_z'].to_numpy().astype(np.float32)
x = np.stack((sc_pos_r, sc_pos_z, ant_pos_z), axis=1)

travel_time = df['travel_time'].to_numpy().astype(np.float32)
path_length = df['path_length'].to_numpy().astype(np.float32)
launch = df['launch_angle'].to_numpy().astype(np.float32)
recieve = df['recieve_angle'].to_numpy().astype(np.float32)
y = np.stack((travel_time, path_length, launch, recieve), axis=1)

x  = x[0::2,:]
x_train = x[:int(x.shape[0]*0.8)]
x_test = x[int(x.shape[0]*0.8):]

y_temp1  = y[0::2,:]
y_temp2  = y[1::2,:]
y = np.zeros((y_temp1.shape[0], 8))

for i in range(4):
    y[:,2*i] = y_temp1[:,i]
    y[:,2*i+1] = y_temp2[:,i]

y_train = y[:int(y.shape[0]*0.8)]
y_test = y[int(y.shape[0]*0.8):]

scaler_x = MinMaxScaler(feature_range=(0,1))
scaler_x.fit(x_train)
norm_x_train = scaler_x.transform(x_train)
norm_x_test = scaler_x.transform(x_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
scaler_y.fit(y_train)
norm_y_train = scaler_y.transform(y_train)
norm_y_test = scaler_y.transform(y_test)





#activation = 'tanh'#keras.layers.Tanh()
#layers = [128, 512, 512, 256, 128]

#model = Fc_model(layers, activation)
#model.build((1,3))

#model = get_simple_model(layers, activation)
#opt = keras.optimizers.Adam(learning_rate=1e-3)
#model.compile(optimizer=opt, loss='mse')
#print(model.summary())


path_of_tuner = '/mnt/md0/aholmberg/models/raytrace_tuner'

if not os.path.isdir(path_of_tuner):
    os.mkdir(path_of_tuner)


tuner = kt.Hyperband(
    model_builder,
    objective='val_loss',
    max_epochs=50,
    directory=path_of_tuner,
    project_name='kt_test'
    )


early_stoping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5
)


tuner.search(
    norm_x_train,
    norm_y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.15,
    callbacks=[early_stoping]
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='/mnt/md0/aholmberg/models/ckpt',
    save_best_only=True,
    save_weights_only=False
)


lr_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5
)


best_hps = tuner.get_best_hyperparameters()[0]


#model = tuner.hypermodel.build(best_hps)
model = model_builder(best_hps)


history = model.fit(norm_x_train, norm_y_train, epochs=50, validation_split=0.15, batch_size=128)

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

model.fit(norm_x_train, norm_y_train, epochs=best_epoch, validation_split=0.15, batch_size=128)

path = '/mnt/md0/aholmberg/models/best_raytrace_model'

if not os.path.isdir(path):
    os.mkdir(path)

model.save(path)


#callback = [model_checkpoint_callback, early_stoping]#, lr_plateau]

#model.fit(norm_x_train, norm_y_train, epochs=50, validation_split=0.15, callbacks=callback)

#y_test = model(norm_x_test).numpy()
