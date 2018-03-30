#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:48:18 2018

@author: radiation
"""


#Python
import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
from PIL import Image
import os


#========= CONFIG FILE TO READ FROM =======
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

#model name
name_experiment = config.get('experiment name', 'name')


#==========predict one image at a time, or predict all images in test folder ==================
test_mode = 'single'

if test_mode == 'single':
    
    test_img_filename = './dataset/test_images/TCGA-HE-7128-01Z-00-DX1.jpeg'
    boundary_filename = './dataset/intBinMask/' + os.path.basename(test_img_filename).split('.')[0] + '_mask_bound.bmp'
    inside_filename = './dataset/intBinMask/' + os.path.basename(test_img_filename).split('.')[0] + '_mask_inside.bmp'
    Imgs_to_test = 1
    
else:
    
    test_imgs_filename = path_data + config.get('data paths', 'test_imgs_original')
    test_masks_filename = path_data + config.get('data paths', 'test_groundTruth')
    #N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    
    test_imgs = load_hdf5(test_imgs_filename)
    test_masks = load_hdf5(test_masks_filename)



#============ Load the data and divide in patches
if test_mode == 'single':
    test_img = np.asarray(Image.open(test_img_filename))
    test_mask = generate_ternary_masks(inside_filename,boundary_filename)
    
    full_img_height,full_img_width = test_img.shape
    
    patches_imgs_test, patches_masks_test = get_data_predict(
        predict_imgs = test_img,
        predict_groundTruth = test_mask,
        patch_height = patch_height,
        patch_width = patch_width,
        N_imgs = Imgs_to_test
    )
    
else:
    patches_imgs_test, patches_masks_test = get_data_predict(
        hdf5_predict_imgs = test_imgs_filename,  #original
        hdf5_predict_groundTruth = test_masks_filename,  #masks
        patch_height = patch_height,
        patch_width = patch_width,
        N_imgs = Imgs_to_test
    )



##================ Run the prediction of the patches ==================================
#best_last = config.get('testing settings', 'best_last')
##Load the saved model
#model = model_from_json(open('./model/' + name_experiment +'_architecture.json').read())
#model.load_weights('./weights/' + name_experiment + '/' + name_experiment + '_'+best_last+'_weights.h5')
#
##Calculate the predictions
#predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
#
#print("predicted images size : {}".format(predictions.shape))
#
##===== Convert the prediction arrays in corresponding images
#boundary_map, inside_map, background_map = pred_to_imgs(predictions, full_img_height, full_img_width)
#
#
#
##========== Elaborate and visualize the predicted images ====================
##visualize boundary_map
##visualize inside_map
##visualize background_map
#
#
##visualize results comparing mask and prediction:
#visualize(boundary_map, './Result/' + name_experiment + '/' + 'predict_boundary_map.jpeg')
#visualize(inside_map, './Result/' + name_experiment + '/' + 'predict_inside_map.jpeg')
#
#
##====== Evaluate the results
#print("\n\n========  Evaluate the results =======================")
#y_scores = predictions
#y_true = test_mask
#
##Area under the ROC curve
#fpr, tpr, thresholds = roc_curve((y_true), y_scores)
#AUC_ROC = roc_auc_score(y_true, y_scores)
## test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
#print("\nArea under the ROC curve: " +str(AUC_ROC))
#
#roc_curve =plt.figure()
#plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
#plt.title('ROC curve')
#plt.xlabel("FPR (False Positive Rate) or 1 - specificity")
#plt.ylabel("TPR (True Positive Rate) or sensitivity")
#plt.legend(loc="lower right")
#plt.savefig('./Result/' + name_experiment + '/' + "ROC.png")

##Precision-recall curve
#precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
#precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
#recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
#AUC_prec_rec = np.trapz(precision,recall)
#print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
#prec_rec_curve = plt.figure()
#plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
#plt.title('Precision - Recall curve')
#plt.xlabel("Recall")
#plt.ylabel("Precision")
#plt.legend(loc="lower right")
#plt.savefig(path_experiment+"Precision_recall.png")
#
##Confusion matrix
#threshold_confusion = 0.5
#print("\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion))
#y_pred =(np.empty((y_scores.shape[0])))
#for i in range(y_scores.shape[0]):
#    if y_scores[i]>=threshold_confusion:
#        y_pred[i]=1
#    else:
#        y_pred[i]=0
#        
#confusion = confusion_matrix(y_true, y_pred)
#print(confusion)
#
#
#accuracy = 0
#if float(np.sum(confusion))!=0:
#    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
#print "Global Accuracy: " +str(accuracy)
#specificity = 0
#if float(confusion[0,0]+confusion[0,1])!=0:
#    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
#print "Specificity: " +str(specificity)
#sensitivity = 0
#if float(confusion[1,1]+confusion[1,0])!=0:
#    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
#print "Sensitivity: " +str(sensitivity)
#precision = 0
#if float(confusion[1,1]+confusion[0,1])!=0:
#    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
#print "Precision: " +str(precision)
#
##Jaccard similarity index
#jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
#print "\nJaccard similarity score: " +str(jaccard_index)
#
##F1 score
#F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
#print "\nF1 score (F-measure): " +str(F1_score)
#
##Save the results
#file_perf = open(path_experiment+'performances.txt', 'w')
#file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
#                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
#                + "\nJaccard similarity score: " +str(jaccard_index)
#                + "\nF1 score (F-measure): " +str(F1_score)
#                +"\n\nConfusion matrix:"
#                +str(confusion)
#                +"\nACCURACY: " +str(accuracy)
#                +"\nSENSITIVITY: " +str(sensitivity)
#                +"\nSPECIFICITY: " +str(specificity)
#                +"\nPRECISION: " +str(precision)
#                )
#file_perf.close()