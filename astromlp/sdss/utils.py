
import os, datetime, requests
import tensorflow as tf
from sklearn.model_selection import train_test_split as sk_train_test_split
import matplotlib.pyplot as plt
import numpy as np

from .datagen import DataGen

def train_test_split(ids):
    IDs_train, IDs_test = sk_train_test_split(ids, train_size=0.75)

    return IDs_train, IDs_test

def train_val_test_split(ids):
    IDs_train, IDs_rest = sk_train_test_split(ids, train_size=0.7)
    IDs_val, IDs_test = sk_train_test_split(IDs_rest, train_size=0.5)

    return IDs_train, IDs_val, IDs_test

def build_datagens(ids, x=None, y=None, batch_size=32, helper=None):
    ids_train, ids_val, ids_test = train_val_test_split(ids)

    train_gen = DataGen(ids_train, x=x, y=y, batch_size=batch_size, helper=helper)
    val_gen = DataGen(ids_val, x=x, y=y, batch_size=batch_size, helper=helper)
    test_gen = DataGen(ids_test, x=x, y=y, batch_size=batch_size, helper=helper)

    return train_gen, val_gen, test_gen

def _moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w

def history_fit_plots(name, history, base_dir='./model_plots', smoothing=False):
    fig = plt.figure(figsize=(18, 6))
    fig.set_facecolor('white')
    plt.ioff()
    curr = 1

    # loss
    loss = history.history['loss']
    if smoothing:
        loss = _moving_average(loss)
    val_loss = history.history['val_loss']
    if smoothing:
        val_loss = _moving_average(val_loss)
    epochs_range = range(1, len(loss)+1)
    plt.subplot(1, 3, curr)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.xticks([x for x in epochs_range if x==1 or x % 5 == 0])
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(visible=True, color='#eeeeee', linestyle='-', linewidth=1)
    curr += 1

    # accuracy
    if 'accuracy' in history.history:
        accuracy = history.history['accuracy']
        if smoothing:
            accuracy = _moving_average(accuracy)
        val_accuracy = history.history['val_accuracy']
        if smoothing:
            val_accuracy = _moving_average(val_accuracy)
        plt.subplot(1, 3, curr)
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.xticks([x for x in epochs_range if x==1 or x % 5 == 0])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.grid(visible=True, color='#eeeeee', linestyle='-', linewidth=1)
        curr += 1

    # mean squared error
    if 'mean_squared_error' in history.history:
        mean_squared_error = history.history['mean_squared_error']
        if smoothing:
            mean_squared_error = _moving_average(mean_squared_error)
        val_mean_squared_error = history.history['val_mean_squared_error']
        if smoothing:
            val_mean_squared_error = _moving_average(val_mean_squared_error)
        plt.subplot(1, 3, curr)
        plt.plot(epochs_range, mean_squared_error, label='Training MSE')
        plt.plot(epochs_range, val_mean_squared_error, label='Validation MSE')
        plt.xlabel('Epochs')
        plt.xticks([x for x in epochs_range if x==1 or x % 5 == 0])
        plt.legend(loc='upper right')
        plt.title('Training and Validation MSE')
        plt.grid(visible=True, color='#eeeeee', linestyle='-', linewidth=1)
        curr += 1

    # mean absolute error
    if 'mean_absolute_error' in history.history:
        mean_absolute_error = history.history['mean_absolute_error']
        if smoothing:
            mean_absolute_error = _moving_average(mean_absolute_error)
        val_mean_absolute_error = history.history['val_mean_absolute_error']
        if smoothing:
            val_mean_absolute_error = _moving_average(val_mean_absolute_error)
        plt.subplot(1, 3, curr)
        plt.plot(epochs_range, mean_absolute_error, label='Training MAE')
        plt.plot(epochs_range, val_mean_absolute_error, label='Validation MAE')
        plt.xlabel('Epochs')
        plt.xticks([x for x in epochs_range if x==1 or x % 5 == 0])
        plt.legend(loc='upper right')
        plt.title('Training and Validation MAE')
        plt.grid(visible=True, color='#eeeeee', linestyle='-', linewidth=1)
        curr += 1

    # save plot
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(os.path.join(base_dir, f'{ name }_plots.png'), bbox_inches='tight')
    plt.close(fig)

def my_callbacks(name=None, path=None, check_point=False, monitor='val_loss', mode='min', tensor_board=True):
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y-%m-%d"))

    my_callbacks = []
    if check_point:
        my_callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path, name),
                                           monitor=monitor, mode=mode,
                                           save_best_only=True, verbose=1))
    if tensor_board:
        my_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    return my_callbacks

