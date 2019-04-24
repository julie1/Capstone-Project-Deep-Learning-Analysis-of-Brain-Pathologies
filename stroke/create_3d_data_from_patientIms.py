#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import sys
import tarfile
import scipy.io as sio
from zipfile import ZipFile as zp

from IPython.display import display, Image
from scipy import ndimage, misc

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split

import h5py
import hdf5storage
import requests
import shutil
from urlparse import urlparse
#from medpy.io import load
import nibabel as nib
import SimpleITK
import json
import random
import shutil


# In[2]:


data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientIms/'
path_test = 'examples/configFiles/deepMedic/test/'
folder = 'U-net_code_stroke/patientIms/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,43), 43)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 


pTest = p[-7:]

for i in range(5):
    pTrain0 = p[:-7]
    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(7*i):] + pTrain0[:-(7*i)]
    
    print (pTrain)
    if i == 0: print(pTest)
    

    train_dict = {'train': ('trainChannels_',path_train,pTrain[0:29]), 'valid': 
                  ('validChannels_',path_valid,pTrain[29:36]), 
                  'test': ('testChannels_',path_test,pTest[:7])}


    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format('brainmask' +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[2]:


data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientIms/'
path_test = 'examples/configFiles/deepMedic/test/'
folder = 'U-net_code_stroke/patientIms/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,43), 43)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 



pTrain0 = p
for i in range(5):

    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(7*i):] + pTrain0[:-(7*i)]
    
    print (pTrain)
 

    train_dict = {'train': ('trainChannels_',path_train,pTrain[0:36]), 'test': 
                  ('testChannels_',path_test,pTrain[36:43])}

    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        path1 = path + str(i) + '/'
       # else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
           # else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            shutil.copytree(src, dst)
           # elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format('brainmask' +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[2]:


#train on all of data in order to do test

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientIms/'
path_test = 'examples/configFiles/deepMedic/test/'
folder = 'U-net_code_stroke/patientIms/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,43), 43)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 


pTest = p[-7:]

for i in range(1):
    pTrain0 = p
    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(7*i):] + pTrain0[:-(7*i)]
    
    print (pTrain)
   # if i == 0: print(pTest)
    

    train_dict = {'train': ('trainChannels_',path_train,pTrain[0:43])}


    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format('brainmask' +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[2]:


#make testing data to upload to Isles2017

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientImsTest/'
path_test = 'examples/configFiles/deepMedic/testIsles2017/'
folder = 'U-net_code_stroke/patientImsTest/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][33:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,32), 32)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 


pTest = p

for i in range(1):
    pTrain0 = p
    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(7*i):] + pTrain0[:-(7*i)]
    
    print (pTrain)
   # if i == 0: print(pTest)
    

    train_dict = {'test': ('testChannels_',path_test,pTrain[0:32])}


    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format('brainmask' +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[2]:


#make training data to upload to Isles2017

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientImsTest/'
path_test = 'examples/configFiles/deepMedic/trainIsles2017/'
folder = 'U-net_code_stroke/patientIms/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,43), 43)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 


pTest = p

for i in range(1):
    pTrain0 = p
    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(7*i):] + pTrain0[:-(7*i)]
    
    print (pTrain)
   # if i == 0: print(pTest)
    

    train_dict = {'test': ('testChannels_',path_test,pTrain[0:43])}


    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format('brainmask' +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[3]:


data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientIms/'
path_test = 'examples/configFiles/deepMedic/test/'
folder = 'U-net_code_stroke/patientIms/'
p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (p_folders)
    

p1 = random.sample(range(0,43), 43)
p = [p_folders[p1[i]]for i in range(len(p_folders))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 



pTest = p[-8:] 
for i in range(5):
   
    pTrain0 = p 
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
   
    else: pTrain = pTrain0[-(8*i):] + pTrain0[:-(8*i)]
    print (pTrain)
    if i == 0: print(pTest)
    


    train_dict = {'train': ('trainChannels_',path_train,pTrain[0:36]), 'valid': 
                      ('validChannels_',path_valid,pTrain[36:43]), 
                      'test': ('testChannels_',path_test,pTest[:7])}


    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[2]:


with open('maskSum.json') as f:
    maskSum = json.load(f)
keys = maskSum.keys()
smallStroke = sorted([int(keys[i]) for i in range(len(keys))if maskSum[keys[i]] < 1000])
largeStroke = sorted([int(keys[i]) for i in range(len(keys))if maskSum[keys[i]] > 1000])
print (smallStroke)
print (largeStroke)


# In[3]:


#small strokes -- less than 1000 one labels in ROI mask region

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
path_train = 'examples/configFiles/deepMedic/train/'
path_valid = 'examples/configFiles/deepMedic/train/validation/'
data_path = 'dataForStrokes/'
imgPath = 'patientIms/'
path_test = 'examples/configFiles/deepMedic/test/'
folder = 'U-net_code_stroke/patientIms/'
#p_folders = [os.path.join(folder, patient) for patient in os.listdir(folder) ]
#p_folders = sorted([int(p_folders[i][29:]) for i in range(len(p_folders))])
print (smallStroke)
    

p1 = random.sample(range(0, 18), 18)
p = [smallStroke[p1[i]]for i in range(len(smallStroke))]
print(p)
#q = [pTrain[-(7*i):] + pTrain[:-(7*i)] for i in range(5)] 


pTest = p[-3:]    
for i in range(5):
    pTrain0 = p[:-3]
    
    
  #  print (pTest)

    if i == 0: pTrain = pTrain0
    else: pTrain = pTrain0[-(3*i):] + pTrain0[:-(3*i)]
    print (pTrain)
    if i == 0: print(pTest)
    

    train_dict = {'train': ('trainChannels_',path_train,pTrain[0:14]), 'valid': 
                  ('validChannels_',path_valid,pTrain[14:17]), 
                  'test': ('testChannels_',path_test,pTest[:3])}
    for key in train_dict:
        if key == 'valid': path_home = '../../../../'
        else: path_home = '../../../'

        name, path, patients = train_dict[key]
        if key != 'test': path1 = path + str(i) + '/'
        else: path1 = path
        tl.files.exists_or_mkdir(path1)
        for datatype in data_types:
            new_file = open(path1+name+datatype+'.cfg', 'w')
            new_file.close()
        if key != 'train':
            new_file1 = open(path1+key+'PredictionNames.cfg', 'w')
            new_file1.close()
        new_file2 = open(path1+key+'ROImasks'+'.cfg', 'w')
        new_file2.close()
        
        for patient in patients: 
            src = os.path.join(folder, str(patient))
            if key != 'test': dst = os.path.join(data_path+key+'/'+str(i)+'/', 'patient_'+str(patient))
            else: dst = os.path.join(data_path+key+'/', 'patient_'+str(patient))
            if key != 'test': shutil.copytree(src, dst)
            elif key == 'test' and i == 0: shutil.copytree(src, dst)
            for datatype in data_types:      
                with open(path1+name+datatype+'.cfg', 'a') as f:            
                    f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                    f.write('\n')
            with open(path1+key+'ROImasks'+'.cfg', 'a') as f:            
                f.write(path_home+dst+'/''{}'.format(datatype +'.nii.gz')) 
                f.write('\n')    
            if key != 'train':
                with open(path1+key+'PredictionNames.cfg', 'a') as f:            
                    f.write('patient_' + str(patient) +'.nii.gz') 
                    f.write('\n') 
           


# In[ ]:




