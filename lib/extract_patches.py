#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:20:05 2018

@author: radiation
"""
import numpy as np
import random
import configparser
import math
from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images


#for training process
def get_data_training(hdf5_train_imgs, hdf5_train_groundTruth, patch_height, patch_width,
                      N_subimgs):
    train_imgs_original = load_hdf5(hdf5_train_imgs)
    
    train_masks = load_hdf5(hdf5_train_groundTruth)
    
    
#    train_imgs = preprocessing(train_imgs_original)
    train_imgs = train_imgs_original
    #extract Training patches from the full images
    
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs)
    
    return patches_imgs_train,patches_masks_train

def paint_border(predict_imgs, patch_height, patch_width):
    [img_h, img_w, channel] = predict_imgs.shape
    new_img_h = img_h + patch_height - 1
    new_img_w = img_w + patch_width - 1
    
    new_data = np.zeros((new_img_h, new_img_w, channel))
    
    new_data[int(patch_height/2):new_img_h-int(patch_height/2),int(patch_width/2):new_img_w-int(patch_width/2),:] = predict_imgs[:,:,:]
    
    return new_data
#for testing process
def get_data_predict(predict_imgs, predict_groundTruth, patch_height, patch_width,
                     N_imgs):
    assert(len(predict_imgs.shape) == 3)
    assert(predict_imgs.shape[2] == 3)
    assert(len(predict_groundTruth.shape) == 3)
   
    [original_h, original_w, channel] = predict_imgs.shape 
    #extend images so that all pixels from original image could be predicted
    predict_imgs = paint_border(predict_imgs, patch_height,patch_width)
    
    
    
    patches_imgs_predict = extract_ordered(predict_imgs[:400,:400,:], patch_height, patch_width)
    
    sample_masks = (predict_groundTruth[:400-patch_height+1,:400-patch_width+1,:] / 127).astype(int)
    patches_masks_predict = np.reshape(sample_masks,[-1,1])
    
    return patches_imgs_predict, patches_masks_predict
    
    

#for prepare dataset
def extract_ordered(full_imgs, patch_h, patch_w):
    '''
    extract patches for image, the mask along with patches
    '''
    num_rows = full_imgs.shape[0] - patch_h + 1
    num_cols = full_imgs.shape[1] - patch_w + 1
    patches = np.empty((num_rows * num_cols, patch_h, patch_w,full_imgs.shape[2]))
    iter_tot = 0
    for row in range(num_rows):
        for col in range(num_cols):
            patches[iter_tot] = full_imgs[row:row + patch_h,col:col + patch_w,:]
            iter_tot += 1
            
    return patches
        
    

def preprocessing(imgs_original):
    '''
    preprocess the image using the method in the paper
    imgs_original: N_imgs * height * width * channel
    
    return: N_imgs * height * width * channel
    '''
    #optical_den = -math.log10(imgs_original / 255)
    pass

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches):
    if N_patches % full_imgs.shape[0] != 0:
        print("please enter a multiple of 24")
        exit()
       
    #check the data consistancy
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[3]==1)   #masks only black and white
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    
    #channel * height * width
    patches = np.empty((N_patches,patch_h,patch_w,full_imgs.shape[3]))
    patches_masks = np.empty((N_patches,patch_h,patch_w,full_masks.shape[3]))
    
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    
    patch_per_img = int(N_patches/full_imgs.shape[0])
    
    iter_tot = 0
    
    patch_per_class = patch_per_img / 3
    for i in range(full_imgs.shape[0]):
        k = 0
        counter = {0:0,1:0,2:0}
        
        while k < patch_per_img:
            
            x_center = random.randint(0+int(patch_w/2),img_w - int(patch_w/2) - 1)
            
            y_center = random.randint(0+int(patch_h/2),img_h - int(patch_h/2) - 1)

            center_label = int(full_masks[i,y_center,x_center,0] / 127)
            
            if counter[center_label] < patch_per_class:
#                print("image:{}, {} class has: {}".format(i,full_masks[i,0,y_center,x_center],counter[full_masks[i,0,y_center,x_center]]))
                patch_img = full_imgs[i,y_center - int(patch_h/2):y_center + int(patch_h/2)+1,x_center - int(patch_w/2):x_center + int(patch_w/2) + 1,:]
                patch_mask = full_masks[i,y_center - int(patch_h/2):y_center + int(patch_h/2)+1,x_center - int(patch_w/2):x_center + int(patch_w/2)+1,:]
                
                patches[iter_tot] = patch_img
                patches_masks[iter_tot] = patch_mask
                
                counter[center_label] += 1
                iter_tot += 1
                k += 1
    
    return patches, patches_masks
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
