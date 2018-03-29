#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:48:18 2018

@author: radiation
"""

#called to predict the input images with fixed NN and trained model and display the result 
#test the model on test dataset
patches_imgs_test, patches_masks_test = get_data_testing(
    hdf5_train_imgs = path_data + config.get('data paths', 'test_imgs_original'),
    hdf5_train_groundTruth = path_data + config.get('data paths', 'test_groundTruth'),
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs =       
) 

score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])