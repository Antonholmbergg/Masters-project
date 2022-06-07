""" For training a single model for the raytracing. Tested wit htensorflow 2.2
"""
import os
from gpuutils import GpuUtils
import raytrace_model_def
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np

GpuUtils.allocate(gpu_count=1, framework='keras')

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


#tf.keras.backend.set_floatx('float64')

"""  # Cartesian/cylindrical coordinates
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
df = pd.read_pickle('/mnt/md0/aholmberg/data/raytracing_random_25_spherical.pkl')

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

# Delete traning points with phi larger than 179 degrees
index = x_train[:, 1] > 179
x_train = np.delete(x_train, index, axis=0)
y_train = np.delete(y_train, index, axis=0)

norm_x_train = np.zeros_like(x_train)
norm_x_train[:, 0] = x_train[:, 0] / (np.sqrt(2700**2 + 2000**2))
norm_x_train[:, 1] = x_train[:, 1] / (180)
norm_x_train[:, 2] = x_train[:, 2] / -(200)

norm_y_train = np.zeros_like(y_train)
norm_y_train[:, 0] = y_train[:, 0] / (20000)
norm_y_train[:, 1] = y_train[:, 1] / (21000)
norm_y_train[:, 2] = y_train[:, 2] / (3500)
norm_y_train[:, 3] = y_train[:, 3] / (3800)
norm_y_train[:, 4] = y_train[:, 4] / (180)
norm_y_train[:, 5] = (y_train[:, 5] - 90) / (90)
norm_y_train[:, 6] = y_train[:, 6] / (180)
norm_y_train[:, 7] = y_train[:, 7] / (90)

depth = 14
width = 50

# Set the name of the model and directory to save the model in
model_name = f'ResNetFc_w{width}_d{depth}_vwidth20_vdepth4_news_lrelu_ln_179deg_250_np'
path = '/mnt/md0/aholmberg/models/' + model_name

opt = keras.optimizers.Adam(learning_rate=5e-4)
act = keras.layers.LeakyReLU()
kernel_init = 'he_normal'

# Define the model
model = raytrace_model_def.get_resnet_model_lrelu(
    width=width,
    depth=depth,
    activation=act,
    kernel_init=kernel_init,
    optimizer=opt,
    )

print(model.summary())

# lr scheduler
def scheduler(epoch, lr):
    if epoch <= 3:
        return 5e-4
    else:
        return lr * tf.math.exp(-0.03)


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# craete the directory for checkpointing models
checkpoint_filepath = '/mnt/md0/aholmberg/models/' + model_name
if not os.path.isdir(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)

# set logs for tensorboard. Performance counters does not work on remote machine with tensorflow<2.5
# logdir = './logs/' + model_name
# profile_batch does not work over ssh with tf version <2.5
# tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq=1)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    save_weights_only=True
)

history = model.fit(
    norm_x_train,
    [norm_y_train[:, :2], norm_y_train[:, 2:4],
     norm_y_train[:, 4:6], norm_y_train[:, 6:]],
    epochs=300,
    batch_size=512,
    validation_split=0.1,
    callbacks=[model_checkpoint_callback, lr_scheduler],  # , tb_callback],
    shuffle=True
    )

# Load and save the best epoch
model.load_weights(checkpoint_filepath)

if not os.path.isdir(path):
    os.mkdir(path)

model.save(path)
# Save loss history
loss_df = pd.DataFrame(history.history)
path_to_loss = 'losses/history_' + model_name + '.pkl'
loss_df.to_pickle(path_to_loss)
