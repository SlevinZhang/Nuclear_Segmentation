#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:44:16 2018

@author: radiation
"""


#1.Normalize color
#2.Generate a ternary mask
#3.divide into pathces(image and mask)
#4.Save result into training,dev and test dataset


#==========================================================
#
#  This prepare the hdf5 datasets of H & E nucleui segementation
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import glob
import sys
sys.path.insert(0, './lib/')
from help_functions import *
import random
#save the data
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#----------------Global parameter ------------------
channels = 3
height = 1000
width = 1000
#---------------------------------------------------
def generate_ternary_masks(inside_mask, boundary_mask):
    '''
    :param mask_name: the filename of mask
    :return:
        a ndarray mask with same size
        '0': represent background
        '1': represent inside
        '2': represent boundary
    '''
    boundary = Image.open(boundary_mask)
    bound_arr = np.asarray(boundary)
#    print("boundary, min:{}, max:{}".format(np.min(bound_arr),np.max(bound_arr)))
    inside = Image.open(inside_mask)
    inside_arr = np.asarray(inside)
#    print("inside, min:{}, max:{}".format(np.min(inside_arr),np.max(inside_arr)))
    
    mask = np.empty((height,width))
    for row in range(height):
        for col in range(width):
            if bound_arr[row,col] == True:
                mask[row,col] = 128
            elif inside_arr[row,col] == True:
                mask[row,col] = 255
            else:
                mask[row,col] = 0
    return mask
def get_datasets(imgs_dir,mask_dir,Nimgs):
    '''
    input:
        imgs_dir: original images
        mask_dir: boundary mask of each image
    output:
        nd-array of imgs
        nd-array of masks(ternary mask)
    '''
    imgs = np.empty((Nimgs,height,width,channels))
    masks = np.empty((Nimgs,height,width))

    image_filenames = glob.glob(imgs_dir + '*.jpeg')
    for index,filename in enumerate(image_filenames):
        basename = os.path.basename(filename)
        print("original image: " + basename)
        img = Image.open(filename)
        imgs[index] = np.asarray(img)

        inside_mask = mask_dir + basename.split('.')[0] + '_mask_inside.bmp'
        boundary_mask = mask_dir + basename.split('.')[0] + '_mask_bound.bmp'
        mask = generate_ternary_masks(inside_mask,boundary_mask)
        masks[index] = mask

    assert(np.max(masks) == 255 and np.min(masks) == 0)

    #reshaping for my standard tensors
#    imgs = np.transpose(imgs,(0,3,1,2))
    
    
    
    masks = np.reshape(masks,(Nimgs,height,width,1))
    assert(imgs.shape == (Nimgs,height,width,channels))
    assert(masks.shape == (Nimgs,height,width,1))

    
    return imgs,masks

#because images don't have the same size, so we also extract patches from it and save patches into hdf5
def get_normalized_datasets(imgs_dir, mask_dir):
    image_filenames = glob.glob(imgs_dir + '*')
    Nimgs = 27000
    patch_w = 51
    patch_h = 51
    
    img_patches = np.empty((Nimgs, patch_h, patch_w, channels))
    
    mask_patches = np.empty((Nimgs, 3))
    
    iter_tot = 0
    for index,filename in enumerate(image_filenames):
        k = 0
        basename = os.path.basename(filename)
        print("original image: " + basename)
        img = Image.open(filename)
        img_w, img_h = img.size
        patch_per_class = 1000
        
        
        mask_filename = mask_dir + 'TM_' + basename.split('.')[0] + '.png'
        mask = Image.open(mask_filename)
        img_array = np.asarray(img)
        mask_array = np.asarray(mask)
        
        counter = {0:0,1:0,2:0}
        
        while k < 3000:
            
            x_center = random.randint(0+int(patch_w/2),img_w - int(patch_w/2) - 1)
            
            y_center = random.randint(0+int(patch_h/2),img_h - int(patch_h/2) - 1)

            # 0 is background
            # 1 is boundary
            # 2 is inside
            center_label = int(mask_array[y_center,x_center]/127)
            
            if counter[center_label] < patch_per_class:
#                print("image:{}, {} class has: {}".format(i,full_masks[i,0,y_center,x_center],counter[full_masks[i,0,y_center,x_center]]))
                patch_img = img_array[y_center - int(patch_h/2):y_center + int(patch_h/2)+1,x_center - int(patch_w/2):x_center + int(patch_w/2) + 1,:]
                
                img_patches[iter_tot] = patch_img
                if  center_label == 0:
                    mask_patches[iter_tot,0]=1
                    mask_patches[iter_tot,1]=0
                    mask_patches[iter_tot,2]=0
                elif center_label == 1:
                    mask_patches[iter_tot,0]=0
                    mask_patches[iter_tot,1]=1
                    mask_patches[iter_tot,2]=0
                    
                else:
                    mask_patches[iter_tot,0]=0
                    mask_patches[iter_tot,1]=0
                    mask_patches[iter_tot,2]=1
                    
                counter[center_label] += 1
                iter_tot += 1
                k += 1
        print(counter)
#    img_patches = np.transpose(img_patches,(0,3,1,2))
    assert(img_patches.shape == (Nimgs,patch_h,patch_w,channels))
    assert(mask_patches.shape == (Nimgs,3))
    return img_patches, mask_patches
    
if __name__ == '__main__':

    dataset_root = "./hdf_dataset/"
    
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
#===============================Get Original dataset ==============================================
    train_images = './dataset/train_images/'
    mask_path = './dataset/intBinMask/'
    
    #Get training dataset
    imgs_train, masks_train = get_datasets(train_images, mask_path,24)
    write_hdf5(imgs_train,dataset_root + 'dataset_imgs_train.hdf5')
    write_hdf5(masks_train,dataset_root + 'dataset_masks_train.hdf5')

#    #getting the same organ testing datasets
#    imgs_test, masks_test = get_datasets(test_images,mask_path,6)
##    print "saving test datasets"
#    write_hdf5(imgs_test,dataset_root + "sameorgan_imgs_test.hdf5")
#    write_hdf5(masks_test, dataset_root + "sameorgan_masks_test.hdf5")
    
    
#=======================Get the normalized dataset from 9 images==============================
#    train_images = './normalized_dataset/images/'
#    mask_path = './normalized_dataset/masks/'
#    
#    imgs_train, masks_train = get_normalized_datasets(train_images,mask_path)
#    write_hdf5(imgs_train, dataset_root + "normalized_dataset_patches_imgs_train.hdf5")
#    write_hdf5(masks_train, dataset_root + "normalized_dataset_patches_masks_train.hdf5")
