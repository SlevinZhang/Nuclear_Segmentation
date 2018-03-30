#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:49:30 2018

@author: radiation
"""

#used to test the trained model

import os
import ConfigParser


#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment!!
name_experiment = config.get('experiment name', 'name')



#check if all of the folder exists
if not os.path.exists('./Result/'+name_experiment):
    print("please train the model at first, then predict")
    exit()
if not os.path.exists('./model/'+name_experiment+'_architecture.json'):
    print("please train the model at first, then predict")
    exit()
    
if not os.path.exists('./weights/' + name_experiment):
    print("please train the model at first, then predict")
    exit()

# finally run the prediction
print("\n2. Run the prediction on GPU (no nohup)")
os.system('python ./src/retinaNN_predict.py')