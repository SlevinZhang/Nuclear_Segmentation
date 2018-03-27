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
                mask[row,col] = 2
            elif inside_arr[row,col] == True:
                mask[row,col] = 1
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

    assert(np.max(masks) == 2 and np.min(masks) == 0)

    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    
    masks = np.reshape(masks,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,channels,height,width))
    assert(masks.shape == (Nimgs,1,height,width))

    
    return imgs,masks


if __name__ == '__main__':
    #------------Path of the images --------------------------------------------------------------
    train_images = './dataset/train_images/'
    mask_path = './dataset/intBinMask/'
    test_images = './dataset/test_images/'
    #---------------------------------------------------------------------------------------------
    dataset_root = "./hdf_dataset/"
    
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    #getting the training datasets
    imgs_train, masks_train = get_datasets(train_images,mask_path,24)
#    print "saving train datasets"
    write_hdf5(imgs_train, dataset_root + "dataset_imgs_train.hdf5")
    write_hdf5(masks_train, dataset_root + "dataset_masks_train.hdf5")

    #getting the same organ testing datasets
    imgs_test, masks_test = get_datasets(test_images,mask_path,6)
#    print "saving test datasets"
    write_hdf5(imgs_test,dataset_root + "sameorgan_imgs_test.hdf5")
    write_hdf5(masks_test, dataset_root + "sameorgan_masks_test.hdf5")
