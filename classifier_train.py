"""Performs a random grid search for the classification model.
Saves all of the models in a specified directory and a file of 
all of the configurations and their best validation accuracy.
 
This code should probably be changed so that multiple models
can be trained in parallel. If running on multiple GPUs then 
set CUDA_VISIBLE_DEVICES='number' when running the code since
it does not pick GPU inteligently. Tested with tensorflow version 2.4
"""
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
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

def get_classifier(width, depth, act, lr, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)):
    """Defines and compiles the classifier model

    Args:
        width (int): The width of all hidden layers in the model.
        depth (int): The number of hidden layers in the model.
        act (str): String name of activation function which can be used in keras layers (e.g. 'relu')
        lr (float): The learning rate for the Adam optimizer
        loss (keras loss function, optional): The loss function to use in the classifier. Defaults to tf.keras.losses.BinaryCrossentropy(from_logits=False).

    Returns:
        _type_: Compiled keras model
    """
    inputs = layers.Input(shape=(3,))
    if act == 'relu':  #  if act is relu this kernel initialisation is better than the default
        k_init = keras.initializers.HeNormal()
    else:
        k_init = keras.initializers.GlorotNormal()

    # define hidden layers
    x = layers.Dense(width, activation=act, kernel_initializer=k_init)(inputs)
    for i in range(depth-1):
        x = layers.Dense(width, activation=act, kernel_initializer=k_init)(x)

    # output layer has sigmoid activation since BinaryCrossentropy(from_logits=False) is used
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()]
        )
    return model


if __name__=="__main__":
    """runs the training of the models
    """
    # load data
    df = pd.read_pickle('/mnt/md0/aholmberg/data/raytracing_class_random_25_spherical.pkl')

    # Set features and labels
    labels = np.array(df['n_sol'])
    sc_pos_d = df['source_pos_d'].to_numpy().astype(np.float32)
    sc_pos_phi = df['source_pos_phi'].to_numpy().astype(np.float32)
    ant_pos_z = df['antenna_pos_z'].to_numpy().astype(np.float32)
    features = np.stack((sc_pos_d, sc_pos_phi, ant_pos_z), axis=1)

    features  = features.astype(np.float32)
    labels[labels == 2] = 1

    # normalize the features
    norm_x = np.zeros_like(features)
    norm_x[:, 0] = features[:, 0] / (np.sqrt(2700**2 + 2000**2))
    norm_x[:, 1] = features[:, 1] / (180)
    norm_x[:, 2] = features[:, 2] / -(200)

    # Split into train and test
    norm_features_train = norm_x[:int(norm_x.shape[0]*0.8)]
    norm_features_test = norm_x[int(norm_x.shape[0]*0.8):]
    labels_train = labels[:int(labels.shape[0]*0.8)]
    labels_test = labels[int(labels.shape[0]*0.8):]

    # Delete data to clear up memory. Might be better to also force the garbage collector.
    del features
    del labels
    del df
    
    # Set the name of the run. will be the name of the directory where the models are saved
    run_name = 'run-1'
    model_dir = f'/mnt/md0/aholmberg/class_models/{run_name}/'
    if not os.path.isdir(model_dir):  # create the directory if it doesn't already exist
        os.mkdir(model_dir)
    # Define the hyperparameters of the random search 
    params = {'lr': [5e-3, 1e-3, 5e-4, 1e-4],
              'depth': [2, 4, 8, 16],
              'width': [24, 48, 96, 128, 256],
              'act': ['relu', 'sigmoid']
              }
    models = []
    for i in range(0, 50):
        # With `clear_session()` called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        tf.keras.backend.clear_session()
        config = {}
        for key in params:  # set the configuration to a random choice from the parameters
            config[key] = random.choice(params[key])
        print(config)

        # get the model with that config and print summary
        model = get_classifier(config['width'], config['depth'], config['act'], config['lr'])
        model.summary()

        # name the model after the configuration parameters
        name = ''
        for key in config:
            name += f'{key}={config[key]}-'
        model_name = f'run{i}-{name[:-1]}'

        # train and save the model
        history = model.fit(norm_features_train, labels_train, epochs=10, batch_size=512, validation_split=0.1, shuffle=True)
        model.save(f'{model_dir}class-{model_name}')
        config['name'] = model_name
        config['val_acc'] = history.history['val_accuracy'].min() # min not tested yet but is a much better idea
        models.append(config)
    
    df = pd.DataFrame(models)
    df.to_pickle(f'class-{run_name}.pkl')  # Might want to cahnge to csv for better compatability, file is small anyway.

    # Fint the best model, evaluate it and print result
    idx = df['val_acc'].idxmax()
    best_model = df['name'].iloc[idx]

    model = keras.models.load_model(f'{model_dir}class-{best_model}')
    y = model.evaluate(norm_features_test, labels_test, batch_size=1024)
    print(y)