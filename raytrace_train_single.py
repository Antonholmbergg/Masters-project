import os
from gpuutils import GpuUtils
from sklearn.preprocessing import MinMaxScaler
import raytrace_model_def
from pickle import dump
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

GpuUtils.allocate(gpu_count=1, framework='keras')

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.set_printoptions(precision=3, suppress=True)


"""  # Cartesian/polar coordinates
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

unique, index, count = np.unique(x, return_counts=True,
                                 return_index=True, axis=0)
x[index[count == 1], :]
x_new = np.delete(x, index[count == 1], axis=0)
y_new = np.delete(y, index[count == 1], axis=0)
unique, index, count = np.unique(x_new, return_counts=True,
                                 return_index=True, axis=0)

 """

# polar/spherical coordinates -------------------------------------------------
df = pd.read_pickle('/mnt/md0/aholmberg/data/raytracing_24_spherical.pkl')

sc_pos_d = df['source_pos_d'].to_numpy().astype(np.float64)
sc_pos_phi = df['source_pos_phi'].to_numpy().astype(np.float64)
ant_pos_z = df['antenna_pos_z'].to_numpy().astype(np.float64)
x = np.stack((sc_pos_d, sc_pos_phi, ant_pos_z), axis=1)

travel_time = df['travel_time'].to_numpy().astype(np.float64)
path_length = df['path_length'].to_numpy().astype(np.float64)
launch = df['l_angle'].to_numpy().astype(np.float64)
recieve = df['r_angle'].to_numpy().astype(np.float64)
y = np.stack((travel_time, path_length, launch, recieve), axis=1)

unique, index, count = np.unique(x, return_counts=True,
                                 return_index=True, axis=0)
x[index[count == 1], :]
x_new = np.delete(x, index[count == 1], axis=0)
y_new = np.delete(y, index[count == 1], axis=0)
unique, index, count = np.unique(x_new, return_counts=True,
                                 return_index=True, axis=0)
# ------------------------------------------------------------------------------

x_train = x_new[0::2, :]

y_temp1 = y_new[0::2, :]
y_temp2 = y_new[1::2, :]
y_new = np.zeros((y_temp1.shape[0], 8))

for i in range(4):
    y_new[:, 2*i] = y_temp1[:, i]
    y_new[:, 2*i+1] = y_temp2[:, i]

y_train = y_new

# scaler_x = MinMaxScaler(feature_range=(0, 1))
# scaler_x.fit(x_train)
# norm_x_train = scaler_x.transform(x_train)
print(x_train.shape)
print(y_train.shape)
norm_x_train = np.zeros_like(x_train)
norm_x_train[:, 0] = x_train[:, 0] / (np.sqrt(2700**2 + 2000**2))
norm_x_train[:, 1] = x_train[:, 1] / (180)
norm_x_train[:, 2] = x_train[:, 2] / -(200)

# scaler_y = MinMaxScaler(feature_range=(0, 1))
# scaler_y.fit(y_train)
# norm_y_train = scaler_y.transform(y_train)

norm_y_train = np.zeros_like(y_train)
norm_y_train[:, 0] = y_train[:, 0] / (20000)
norm_y_train[:, 1] = y_train[:, 1] / (21000)
norm_y_train[:, 2] = y_train[:, 2] / (3500)
norm_y_train[:, 3] = y_train[:, 3] / (3800)
norm_y_train[:, 4] = y_train[:, 4] / (180)
norm_y_train[:, 5] = (y_train[:, 5] - 90) / (90)
norm_y_train[:, 6] = y_train[:, 6] / (180)
norm_y_train[:, 7] = y_train[:, 7] / (90)

depth = 4
width = 60

model_name = f'ResNetFc_w{width}_d{depth}_vwidth1_news_selu_ln'
path = '/mnt/md0/aholmberg/models/' + model_name

scaler_path_x = path + "-x-scaler.pkl"
scaler_path_y = path + "-y-scaler.pkl"

#dump(scaler_x, open(scaler_path_x, 'wb'))
#dump(scaler_y, open(scaler_path_y, 'wb'))

opt = keras.optimizers.Adam(learning_rate=1e-3)
# act = layers.LeakyReLU(alpha=0.3)
act = 'selu'
kernel_init = keras.initializers.lecun_normal()
ln = True
model = raytrace_model_def.get_resnet_model_new(
    width=width,
    depth=depth,
    activation=act,
    kernel_init=kernel_init,
    ln=ln,
    optimizer=opt,
    )

print(model.summary())


def scheduler(epoch, lr):
    if epoch <= 3:
        return 1e-3
    else:
        return lr * tf.math.exp(-0.05)


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = '/mnt/md0/aholmberg/models/' + model_name
if not os.path.isdir(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)

logdir = './logs/' + model_name
# profile_batch='1000,1020') does not work over ssh with tf version <2.5
tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq=1)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    save_weights_only=True
)

history = model.fit(
    norm_x_train,
    [norm_y_train[:, :2], norm_y_train[:, 2:4],
     norm_y_train[:, 4:6], norm_y_train[:, 6:]],
    epochs=180,
    batch_size=512,
    validation_split=0.1,
    callbacks=[model_checkpoint_callback, lr_scheduler, tb_callback],
    shuffle=True
    )

model.load_weights(checkpoint_filepath)

if not os.path.isdir(path):
    os.mkdir(path)

model.save(path)

loss_df = pd.DataFrame(history.history)
path_to_loss = 'losses/history_' + model_name + '.pkl'
loss_df.to_pickle(path_to_loss)
