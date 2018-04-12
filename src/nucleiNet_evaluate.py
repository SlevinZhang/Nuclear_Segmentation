# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:31:05 2018

@author: siliangzhang
"""
from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import configparser

#================ Run the prediction of the patches ==================================
config = configparser.RawConfigParser()
config.read('configuration.txt')

#model name
name_experiment = config.get('experiment name', 'name')

best_last = config.get('testing settings', 'best_last')
#Load the saved model
model = model_from_json(open('./model/' + name_experiment +'_architecture.json').read())
model.load_weights('./weights/' + name_experiment + '/' + name_experiment + '_'+best_last+'_weights.h5')

dev_datagen = ImageDataGenerator()
dev_datagen = dev_datagen.flow_from_directory(
        './dataset/dev_patches/',
        target_size = (51,51),
        class_mode = 'categorical',
        batch_size = 128,
        )

metrics = model.evaluate_generator(dev_datagen,steps = 375,verbose=1)
