#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:23:58 2018

@author: radiation
"""

import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


#     print(region.attrib['Id'])
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import glob
import os
import h5py

#for training process
def get_data_training(hdf5_train_imgs, hdf5_train_groundTruth, patch_height, patch_width,
                      N_subimgs):
    train_imgs_original = load_hdf5(hdf5_train_imgs)
    
    train_masks = load_hdf5(hdf5_train_groundTruth)
    
    
    train_imgs = preprocessing(train_imgs_original)
    #extract Training patches from the full images
    
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs)
    
    return patches_imgs_train,patches_masks_train

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches):
    if N_patches % full_imgs.shape[0] != 0:
        print("please enter a multiple of 24")
        exit()
       
    #check the data consistancy
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    
    #channel * height * width
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    
    patch_per_img = int(N_patches/full_imgs.shape[0])
    
    iter_tot = 0
    
    for i in range(full_imgs.shape[0]):
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w - int(patch_w/2))
            
            y_center = random.randint(0+int(patch_h/2),img_h - int(patch_h/2))
            
            patch_img = full_imgs[i,:,y_center - int(patch_h/2):y_center + int(patch_h/2),x_center - int(patch_w/2):x_center + int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center - int(patch_h/2):y_center + int(patch_h/2),x_center - int(patch_w/2):x_center + int(patch_w/2)]
            
            patches[iter_tot] = patch_img
            patches_masks[iter_tot] = patch_mask
            
            iter_tot += 1
            k += 1
    
    return patches, patches_masks

def group_images(data,per_row):
    assert data.shape[0]% per_row == 0
    
    assert (data.shape[3]==1 or data.shape[3]==3)
    
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def visualize(data,filename):
    
    assert (len(data.shape)==3) # height*width*channels
    img = None
    if data.shape[2]==1:  # in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img

def parse_mask(mask):
    assert(len(mask.shape)==2)
    inside_masks = mask.copy()
    inside_masks[mask==2] = 0
    boundary_masks = mask.copy()
    boundary_masks[mask==1] = 0
    boundary_masks[mask==2] = 1

    return inside_masks, boundary_masks
#---------------------Global Variable-----------------
hdf_train_imgs = './hdf_dataset/dataset_imgs_train.hdf5'
hdf_train_masks = './hdf_dataset/dataset_masks_train.hdf5'
root_visualize = './Result/visualize/'

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

train_img_patches, train_mask_patches = get_data_training(hdf_train_imgs,hdf_train_masks,
                                                          51,51,342000)

img_patches_group = group_images(train_img_patches[:25],5)
mask_patches_group = group_images(train_mask_patches[:25],5)

inside_masks, boundary_masks = parse_mask(mask_patches_group)

visualize(img_patches_group,root_visualize + 'img_patches').show()
visualize(inside_masks,root_visualize + 'inside_mask_patches').show()
visualize(boundary_masks,root_visualize + 'boundary_mask_patches').show()




#-------------------------------------------------------------------------------------
#print("The shape of group masks: {}".format(image_group.shape))
#print("original min: {}, max:{}".format(np.min(image_group),np.max(image_group)))
#inside_masks[image_group==2] = 0
#print("inside min: {}, max:{}".format(np.min(inside_masks),np.max(inside_masks)))
##visualize data
#visualize(inside_masks,root_visualize + 'inside_mask').show()
#
#
#boundary_masks[image_group==1] = 0
#boundary_masks[image_group==2] = 1
#print("min: {}, max:{}".format(np.min(boundary_masks),np.max(boundary_masks)))
#visualize(boundary_masks,root_visualize+'boundary_mask').show()
