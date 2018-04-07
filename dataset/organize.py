# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:36:21 2018

@author: siliangzhang
"""

import glob
import os
import shutil
from PIL import Image
import numpy as np

train_dataset = './train/'
train_mask_dataset = './train_mask/'
test_same_organ = './test_same_organ/'
test_same_organ_mask = './test_same_organ_mask/'
test_diff_organ = './test_diff_organ/'
test_diff_organ_mask = './test_diff_organ_mask/'
mask_root = './intBinMask/'
patches_root = './train_patches/'
#======================================================divide into pathces ===========================================

if not os.path.exists(patches_root):
    os.mkdir(patches_root)

train_imgs = glob.glob(train_dataset + '*.jpg')
index = 0
for filename in train_imgs:
    counter = {0:0,1:0,2:0}
    basename = os.path.basename(filename).split('.')[0]
    boundary_mask = Image.open(train_mask_dataset + basename + '_mask_bound.bmp')
    inside_mask = Image.open(train_mask_dataset + basename + '_mask_inside.bmp')
    img = Image.open(filename)
    img_arr = np.asarray(img)
    bound_arr = np.asarray(boundary_mask)
    inside_arr = np.asarray(inside_mask)
    [height, width,chal] = np.shape(img_arr)
    patch_width = 51
    patch_height = 51
    for row in range(0+int(patch_height/2),height-int(patch_height/2)-1):
        for col in range(0+int(patch_width/2),width-int(patch_width/2)-1):
            if bound_arr[row,col]:
                patch_label = 2
            elif inside_arr[row,col]:
                patch_label = 1
            else:
                patch_label = 0
            if counter[patch_label] < 10000:
                patch_img = img_arr[row -int(patch_height/2):row + int(patch_height/2)+1,col - int(patch_height/2):col + int(patch_width/2) + 1,:]
                patch = Image.fromarray(patch_img)
                patch.save(patches_root + str(patch_label) + '/' + str(index) + '.jpg')
                counter[patch_label] += 1
                index += 1
                print(counter)

#========================================================training set statistics======================================
#train_boundary_files = glob.glob(train_mask_dataset + '*_mask_bound.bmp')
#train_inside_files = glob.glob(train_mask_dataset + '*_mask_inside.bmp')
#
#train_bound_distribution = []
#train_inside_distribution = []
#for filename in train_boundary_files:
#    img = Image.open(filename)
#    img_arr = np.asarray(img)
##    print("min:{},max:{}".format(np.min(img_arr),np.max(img_arr)))
#    num_points = np.sum(img_arr)
#    print(num_points)
#    train_bound_distribution.append(num_points)
#
#for filename in train_inside_files:
#    img = Image.open(filename)
#    img_arr = np.asarray(img)
#    num_points = np.sum(img_arr)
#    print(num_points)
#    train_inside_distribution.append(num_points)
#
#print("boundary distribution: {}".format(train_bound_distribution))
#print("inside distribution: {}".format(train_inside_distribution ))
#
#print("total boundary points: {}".format(np.sum(train_bound_distribution)))
#print("total inside points: {}".format(np.sum(train_inside_distribution)))
#=====================================================Split train and test dataset==================================
#train_files = glob.glob(train_dataset + '*.jpg')
#
#for filename in train_files:
#    basename = os.path.basename(filename).split('.')[0]
#    print(basename)
#    src_boundary_mask = mask_root + basename + '_mask_bound.bmp'
#    dst_boundary_mask = train_mask_dataset +basename + '_mask_bound.bmp'
#    src_inside_mask = mask_root + basename + '_mask_inside.bmp'
#    dst_inside_mask = train_mask_dataset + basename + '_mask_inside.bmp' 
#    
#    shutil.copy(src_boundary_mask,dst_boundary_mask)
#    shutil.copy(src_inside_mask, dst_inside_mask)
#
#
#test_same_files = glob.glob(test_same_organ + '*.jpg')
#
#for filename in test_same_files:
#    basename = os.path.basename(filename).split('.')[0]
#    print(basename)
#    src_boundary_mask = mask_root + basename + '_mask_bound.bmp'
#    dst_boundary_mask = test_same_organ_mask +basename + '_mask_bound.bmp'
#    src_inside_mask = mask_root + basename + '_mask_inside.bmp'
#    dst_inside_mask = test_same_organ_mask + basename + '_mask_inside.bmp' 
#    
#    shutil.copy(src_boundary_mask,dst_boundary_mask)
#    shutil.copy(src_inside_mask, dst_inside_mask)
#
#
#test_diff_files = glob.glob(test_diff_organ + '*.jpg')
#
#for filename in test_diff_files:
#    basename = os.path.basename(filename).split('.')[0]
#    print(basename)
#    src_boundary_mask = mask_root + basename + '_mask_bound.bmp'
#    dst_boundary_mask = test_diff_organ_mask +basename + '_mask_bound.bmp'
#    src_inside_mask = mask_root + basename + '_mask_inside.bmp'
#    dst_inside_mask = test_diff_organ_mask + basename + '_mask_inside.bmp' 
#    
#    shutil.copy(src_boundary_mask,dst_boundary_mask)
#    shutil.copy(src_inside_mask, dst_inside_mask)
    
    
#================================parse the name of images and divide it into different category===================
#data_root = './normalized_IEEE_1/'
#filenames = glob.glob(data_root + '*.jpg')
##for filename in filenames:
##    new_name = os.path.basename(filename).split('.')[0] + '.jpg'
##    print(new_name)
##    os.rename(filename,data_root + new_name)
#cate_dict = {
#        '18':'lung',
#        '21':'lung',
#        '38':'lung',
#        '49':'lung',
#        '50':'lung',
#        'A7':'breast',
#        'AR':'breast',
#        'AY':'colon',
#        'B0':'kidney',
#        'CH':'prostate',
#        'DK':'bladder',
#        'E2':'breast',
#        'G2':'bladder',
#        'G9':'prostate',
#        'HE':'kidney',
#        'KB':'stomach',
#        'NH':'colon',
#        'RD':'stomach'
#        }
#for filename in filenames:
#    category = os.path.basename(filename).split('-')[1]
#    new_root = data_root + cate_dict[category] + '/'
#    if not os.path.exists(new_root):
#        os.mkdir(new_root)
#    os.rename(filename, new_root + os.path.basename(filename))
    