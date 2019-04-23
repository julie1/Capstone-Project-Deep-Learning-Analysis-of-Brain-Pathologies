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
from medpy.io import load
from medpy.io import save
import SimpleITK
import json


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages(dataset,m, n): # shows size of the sample    
    
   # indices=np.random.choice(dataset.shape[0], n)
    fig=plt.figure()   
    for i in range(m, m+2*n):
        a=fig.add_subplot(2,n,i+1-m)
        if i < n/2 : d = dataset[i,:,:]
        else: d = dataset[:,:,i]
        plt.imshow(d)
        
        # a.set_title(chr(labels[indices[i]]+ord('A')))
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        
    plt.show()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages1(dataset,n): # shows size of the sample    
    
   # indices=np.random.choice(dataset.shape[0], n)
    fig=plt.figure()   
    for i in range(n):
        a=fig.add_subplot(1,n,i+1)
        d = dataset[:, :, i]
        
        plt.imshow(d)
        
        # a.set_title(chr(labels[indices[i]]+ord('A')))
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        
    plt.show()


# In[6]:


mri_types = ['Flair', 'T1', 'T1c', 'T2', 'OT']
pathHGG = "/home/julie/U-net_code_Tumor/BRATS2015_Training/BRATS2015_Training/HGG/"
pathLGG = "/home/julie/U-net_code_Tumor/BRATS2015_Training/BRATS2015_Training/LGG/"
folderHGG = [os.path.join(pathHGG, patient) for patient in os.listdir(pathHGG)][0]
folderLGG = [os.path.join(pathLGG, patient) for patient in os.listdir(pathLGG)][0]
print('HGG:')
for mri_type in mri_types:
    patient = sorted(glob(os.path.join(folderHGG, 'VSD*'+ mri_type+'.*')))            
    mri_name = glob(os.path.join(patient[0], 'VSD*'))[0]
    im = load(mri_name)[0]
    print(mri_type+':')
    showDataImages(im,80,3)
    
print('LGG:')
for mri_type in mri_types:
    patient = sorted(glob(os.path.join(folderLGG, 'VSD*'+ mri_type+'.*')))            
    mri_name = glob(os.path.join(patient[0], 'VSD*'))[0]
    im = load(mri_name)[0]
    print(mri_type+':')
    showDataImages(im,100,3)


# In[ ]:




