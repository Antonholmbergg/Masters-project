from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow import keras
from radiotools import plthelpers as php
from sklearn.externals import joblib

df = pd.read_csv('/mnt/md0/aholmberg/data/raytrace_samples_random.csv')

sc_pos_r = df['source_pos_r'].to_numpy().astype(np.float32)
sc_pos_z = df['source_pos_z'].to_numpy().astype(np.float32)
ant_pos_z = df['antenna_pos_z'].to_numpy().astype(np.float32)
x = np.stack((sc_pos_r, sc_pos_z, ant_pos_z), axis=1)

travel_time = df['travel_time'].to_numpy().astype(np.float32)
path_length = df['path_length'].to_numpy().astype(np.float32)
launch = df['launch_angle'].to_numpy().astype(np.float32)
recieve = df['recieve_angle'].to_numpy().astype(np.float32)
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

model_path = '/mnt/md0/aholmberg/models/raytrace_6x1024-4-out'
path_of_plot = '/mnt/md0/aholmberg/plots/raytrace/raytrace_6x1024-4-out-new-scaling/'


scaler_x = joblib.load(model_path + '-x-scaler.save')
scaler_y = joblib.load(model_path + '-y-scaler.save')

x_test  = x_new[0::2,:]


y_temp1  = y_new[0::2,:]
y_temp2  = y_new[1::2,:]
y_test = np.zeros((y_temp1.shape[0], 8))

for i in range(4):
    y_test[:,2*i] = y_temp1[:,i]
    y_test[:,2*i+1] = y_temp2[:,i]


norm_x_test = scaler_x.transform(x_test)
norm_y_test = scaler_y.transform(y_test)


model = keras.models.load_model(model_path)
print(model.summary())


#y_test_pred = model(norm_x_train).numpy()
temp1 = model(norm_x_test)
y_test_pred = np.concatenate((temp1[0], temp1[1], temp1[2], temp1[3]), axis=1)


y_test_inv = scaler_y.inverse_transform(y_test_pred)


#diff_deg = y_train - y_test_inv
diff_deg = y_test - y_test_inv
#diff = y_train - y_test_inv
#diff_deg = np.copy(diff)
#diff_deg[:,4:] = np.degrees(diff_deg[:, 4:])


sol = ['time_sol_1:', 
       'time_sol_2:',
       'length_sol_1:',
       'length_sol_2:',
       'launch_sol_1:',
       'launch_sol_2:',
       'recieve_sol_1:',
       'recieve_sol_2:']



if not os.path.isdir(path_of_plot):
    os.mkdir(path_of_plot)

for i in range(8):
    #mean = np.mean(diff[:,i])
    #std = np.std(diff[:,i])
    mean = np.mean(diff_deg[:,i])
    std = np.std(diff_deg[:,i])
    print(sol[i] + f' mean: {mean:.4f}  std: {std:.4f}')
    fix, ax = php.get_histogram(diff_deg[:,i], bins=50)
    plt.savefig(path_of_plot + sol[i][:-1] + '.png')