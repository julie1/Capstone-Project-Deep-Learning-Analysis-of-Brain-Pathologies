#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import nibabel as nib
import os, time
from glob import glob
import sys
import tarfile
import scipy.io as sio
from zipfile import ZipFile as zp

from IPython.display import display, Image
from scipy import ndimage, misc
import surface_distance as sd


from scipy.stats import uniform
from scipy.ndimage import morphology

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from sklearn.model_selection import RandomizedSearchCV as rand

import h5py
import hdf5storage
import requests
import shutil
from urlparse import urlparse
from medpy.io import load
import SimpleITK
import json


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages(dataset,n): # shows size of the sample    
    
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


# In[3]:


IslesCRFpath = 'examples/outputIsles2017/'
CRFpath = 'outputTrain/outputCRFtrain/'
pathG = 'dataForStrokes/test/'
path_output = IslesCRFpath+CRFpath
path_ground = IslesCRFpath+pathG
def get_patient(p):    
    l1 = p.find('patient_')
    l2 = p.find('_CRF')
   # print(p[l1+8:])
    return int(p[l1+8:l2]) if l2!=-1 else int(p[l1+8:])
patients = sorted(os.listdir(path_output), key=get_patient) 
grounds = sorted(os.listdir(path_ground), key=get_patient)
p_folders = [os.path.join(path_output, patient) for patient in patients]
g_folders = [os.path.join(path_ground, patient) for patient in grounds]
#print(p_folders)
#print(g_folders)


# In[4]:


print("Segmentation", "             Ground")
for i in range(len(p_folders)):
    print ('patient:', i)
    im1 = nib.load(p_folders[i]).get_data()
    im2 = nib.load(os.path.join(g_folders[i], 'OT.nii.gz')).get_data()
    j = np.argmax(np.sum(im2, (0,1)))
   # print(j, np.sum(im2))
    im = np.stack([im1[:,:,j],im2[:,:,j]], axis=2)
    print(im.shape)
    showDataImages(im,2)
   # showDataImages1(im1,1, j)
   # showDataImages1(im2,1, j)


# In[ ]:





# In[ ]:




