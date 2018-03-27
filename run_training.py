#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:43:22 2018

@author: radiation
"""

#used to train the model

import os, sys

import configparser


#config file to read from
config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))

#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results
result_dir = './Result/' + name_experiment

print("\n1. Create directory for the results (if not already existing)")

if os.path.exists(result_dir):

    print("Dir already existing")

elif sys.platform=='win32':

    os.system('mkdir ' + result_dir)

else:

    os.system('mkdir -p ' +result_dir)

print("copy the configuration file in the results folder")

if sys.platform=='win32':

    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')

else:

    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')


print("\n2. Run the training on GPU")
os.system(run_GPU +' python ./src/nucleiNet_training.py')

