#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:47:46 2018

@author: radiation
"""

#called by training process, generate the neural network and save the neural network and trained model

###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training process
#
##################################################


import numpy as np
import configparser

from keras import optimizers
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, core, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import vgg16

import sklearn.preprocessing as press
import tensorflow as tf

import sys
import os
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

#a new metric method
def aji(y_predicts, y_groundtruth):
    '''
    do the calculation and return the score
    '''
    pass

def get_resNet(n_ch, patch_height, patch_width, learning_rate):
    base_model = vgg16.VGG16(include_top=False,input_shape=(patch_height,patch_width,n_ch))
    last = base_model.layers[-1].output
    print("output: {}".format(last))
    flat = Flatten()(last)
    print("Flatten: {}".format(flat))
    fc1 = Dense(1024,activation='relu')(flat)
    fc2 = Dense(1024,activation='relu')(fc1)

    predictions = Dense(3,activation='softmax')(fc2)
    
    model = Model(input = base_model.inputs, output = predictions)
    
    model.compile(optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9,
                                                beta_2=0.99),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model
    
#Define the neural network
def get_nucleiNet(n_ch,patch_height,patch_width,learning_rate):
    
    inputs = Input(shape=(patch_height,patch_width,n_ch))
    conv1 = Conv2D(25, (4, 4), activation='relu', padding='same',data_format='channels_last')(inputs)
    #conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D((2, 2),data_format='channels_last')(conv1)
    #
    conv2 = Conv2D(50, (5, 5), activation='relu', padding='same',data_format='channels_last')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2),data_format='channels_last')(conv2)
    #
    conv3 = Conv2D(80, (6, 6), activation='relu', padding='same',data_format='channels_last')(pool2)
    #conv3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling2D((2,2),data_format='channels_last')(conv3)
    
    #Flatten out
    pool3 = Flatten()(pool3)
    
    fc1 = Dense(1024, activation='relu')(pool3)
    #fc1 = Dropout(0.5)(fc1)
    
    fc2 = Dense(512, activation='relu')(fc1)
    #fc2 = Dropout(0.5)(fc2)
    
    fc3 = Dense(3, activation='softmax')(fc2)

    model = Model(input=inputs, output=fc3)

#    #Adam optimizer
#    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy',metrics=['accuracy'])
#    
    #SGD optimizer
    sgd = SGD(lr=0.001, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

#========= check if the data format is channel first ================
if K.image_data_format() != 'channels_last':
    K.set_image_data_format('channels_last')
    assert(K.image_data_format() == 'channels_last')

#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
learning_rate = float(config.get('training settings', 'learning_rate'))
N_subimgs = int(config.get('training settings', 'N_subimgs'))


#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    hdf5_train_imgs = path_data + config.get('data paths', 'train_imgs_original'),
    hdf5_train_groundTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = N_subimgs
)


#===============================Load in normaized data======================================
#patches_imgs_train = load_hdf5('./hdf_dataset/normalized_dataset_patches_imgs_train.hdf5')
#patches_masks_train = load_hdf5('./hdf_dataset/normalized_dataset_patches_masks_train.hdf5')




#========= Save and visualize a sample of what you're feeding to the neural network ==========
N_sample = 40
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./Result/'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./Result/'+name_experiment+'/'+"sample_input_mask")#.show()


#=========== Construct and save the model arcitecture =====
patch_height = patches_imgs_train.shape[1]
patch_width = patches_imgs_train.shape[2]
n_ch = patches_imgs_train.shape[3]


#model = get_nucleiNet(n_ch, patch_height, patch_width,learning_rate)  #the nucleiNet model
#model = get_resNet(n_ch, patch_height,patch_width,learning_rate) # the resNet model
model = get_resNet(3,51,51,learning_rate)

print(model.summary())


#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()

#save the architecture of model
open('./model/'+name_experiment +'_architecture.json', 'w').write(json_string)

#============  Training ==================================
if os.path.exists('./weights/' + name_experiment + '/'):
    #clear the old files
    os.system('rm -rf ./weights/'+name_experiment + '/')    
    print("Dir already existing")

if sys.platform=='win32':

    os.system('mkdir ' + './weights/' + name_experiment + '/')

else:
    os.system('mkdir -p ' + './weights/' + name_experiment + '/')
    


#==============Calculate class distribution in patches===========================
patches_masks_train = masks_nucleiNet(patches_masks_train)
class_distribution_train(patches_masks_train)


print("Done with parse masks")
#train the model with number of batches and number of batch_size

##========== Data augmentation =============================
#datagen = ImageDataGenerator(
#        horizontal_flip = True,
#        vertical_flip = True,
#        data_format='channels_first')
#train_datagen = ImageDataGenerator(
#        horizontal_flip = True,
#        vertical_flip = True
#        )
#train_datagen = train_datagen.flow_from_directory(
#        './dataset/train_patches/',
#        target_size=(51,51),
#        class_mode = 'categorical',
#        batch_size = 128,
#        )
#dev_datagen = ImageDataGenerator()
#       
#dev_datagen = dev_datagen.flow_from_directory(
#        './dataset/dev_patches/',
#        target_size = (51,51),
#        class_mode = 'categorical',
#        batch_size = 128,
#        )
#========= Training =====================================


#only save the best model performance on validation set
checkpointer = ModelCheckpoint(filepath='./weights/' + name_experiment + '/' +name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
#early stop criteria, if the val_acc doesn't change after 10 epoches
earlyStopping = EarlyStopping(monitor='val_loss',patience=5)


History = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, 
          batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, 
          callbacks=[checkpointer,earlyStopping])
#model.fit_generator(train_datagen, steps_per_epoch = 3375, epochs=N_epochs, verbose = 1,
#                             validation_data = dev_datagen,
#                             validation_steps = 375,
#                             shuffle=True,
#			     callbacks=[checkpointer, earlyStopping]
#				)
#
#========== Save and test the last model ===================
model.save_weights('./weights/' + name_experiment + '/' + name_experiment + '_last_weights.h5', overwrite=True)


#print(History.history)




















#
