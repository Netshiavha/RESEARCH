from numpy.random import seed
seed(12345)

import tensorflow as tf
tf.random.set_seed(1234)

import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from utils import DataGenerator
from unet3 import *  # Ensure that your UNet model definition is in unet3.py

def main():
    goTrain()

def goTrain():
    # Input image dimensions
    params = {
        'batch_size': 1,
        'dim': (128, 128, 128),
        'n_channels': 1,
        'shuffle': True
    }
    seismPathT = "/content/drive/MyDrive/faultSeg-master/faultSeg-master/data/train/seis/"
    faultPathT = "/content/drive/MyDrive/faultSeg-master/faultSeg-master/data/train/fault/"

    seismPathV = "/content/drive/MyDrive/faultSeg-master/faultSeg-master/data/validation/seis/"
    faultPathV = "/content/drive/MyDrive/faultSeg-master/faultSeg-master/data/validation/fault/"
    train_ID = range(200)
    valid_ID = range(20)

    # Data Generators
    train_generator = DataGenerator(dpath=seismPathT, fpath=faultPathT, data_IDs=train_ID, **params)
    valid_generator = DataGenerator(dpath=seismPathV, fpath=faultPathV, data_IDs=valid_ID, **params)

    # Model Initialization
    model = unet(input_size=(None, None, None, 1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Checkpoint
    filepath = "/content/drive/MyDrive/faultSeg-master/faultSeg-master/check2/fseg-{epoch:02d}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
    logging = TrainValTensorBoard()

    # Callbacks list
    callbacks_list = [checkpoint, logging]
    print("Data prepared, ready to train!")

    # Fit the model
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15,
        callbacks=callbacks_list,
        verbose=1
    )

    model.save('/content/drive/MyDrive/faultSeg-master/faultSeg-master/check1/fseg.keras')
    showHistory(history)

def showHistory(history):
    # Summarize history for accuracy and loss in subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='train', color='blue')
    ax1.plot(history.history['val_accuracy'], label='test', color='orange')
    ax1.set_title('Model accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Loss plot
    ax2.plot(history.history['loss'], label='train', color='blue')
    ax2.plot(history.history['val_loss'], label='test', color='orange')
    ax2.set_title('Model loss', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)


    save_path = '/content/drive/MyDrive/faultSeg-master/faultSeg-master/South Deep_3D/model_performance.pdf'
    plt.savefig(save_path, format='pdf')

    plt.tight_layout()
    plt.show()

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log1', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)

    def set_model(self, model):
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        with self.val_writer.as_default():
            for name, value in val_logs.items():
                tf.summary.scalar(name, value, step=epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        logs.update({'lr': tf.keras.backend.get_value(self.model.optimizer.learning_rate)})
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':
    main()


