# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:25:05 2021

@author: conor
"""

import numpy as np
import tensorflow as tf
import ketos.data_handling.database_interface as dbi
from ketos.neural_networks.resnet import ResNetInterface
from ketos.data_handling.data_feeding import BatchGenerator
#import packages

db = dbi.open_file("/content/drive/MyDrive/Colab Notebooks/TRAIN/database.h5", 'r')
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/val/data")
#open connection to database and tables

def transform_batch(X, Y):
  x = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
  y = tf.one_hot(Y['label'], depth=2, axis=1).numpy()
  return x, y
train_generator = BatchGenerator(batch_size=128, data_table=train_data, 
                                  output_transform_func=ResNetInterface.transform_batch,
                                  shuffle=True, refresh_on_epoch_end=True)
val_generator = BatchGenerator(batch_size=128, data_table=val_data,
                                 output_transform_func=ResNetInterface.transform_batch,
                                 shuffle=True, refresh_on_epoch_end=False)
#create datafeed

resnet = ResNetInterface.build_from_recipe_file("/content/drive/MyDrive/Colab Notebooks/TRAIN/recipe.json")
resnet.train_generator = train_generator
resnet.val_generator = val_generator
resnet.checkpoint_dir = "checkpoints"
#create neural network

resnet.train_loop(n_epochs=30, verbose=True)
resnet.save_model('FW20.kt',audio_repr_file="/content/drive/MyDrive/Colab Notebooks/TRAIN/spec_config.json")
#train and save the model