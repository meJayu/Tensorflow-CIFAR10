# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:18:24 2017

Simple handler to get the image data from the directory.

@author: jay
"""

import os
from skimage import data


ROOT_PATH = '/home/jay/Dataset'
traindata_path = os.path.join(ROOT_PATH,'BelgiumTSC_Training/Training/')
testdata_path = os.path.join(ROOT_PATH,"BelgiumTSC_Testing/Testing/")

def get_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]               
    images = []
    labels = []

    for d in directories:
        label_directory = os.path.join(data_directory,d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
                            
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels
                    