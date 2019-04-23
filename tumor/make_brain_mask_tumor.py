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
import nilearn
from medpy.io import load
import SimpleITK as sitk

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
from nilearn.masking import compute_multi_epi_mask, compute_epi_mask 
from nilearn.masking import compute_multi_background_mask 


# In[2]:


mri_types = ['Flair', 'T1', 'T1c', 'T2', 'OT']
path = "BRATS2015_Training/BRATS2015_Training/"
nii_path = 'Training_nii/'
grades = ['HGG', 'LGG']


def get_patient_num(p):
    i1 = p.find('pat')
    
    num = p[i1:]
    return num

def make_patients(grade):
    num_patients = len(os.listdir(path+grade)) 
    num = num_patients
    print(num)
    p_folders = [os.path.join(path+grade, patient) for patient in os.listdir(path+grade) ] 
  #  print(p_folders[0])    
    patients = [get_patient_num(p_folders[i])for i in range(len(p_folders))]
  #  print(patients)
    #create patient image path lists for each type of mri
    return p_folders, patients

def mha_to_nii(patient,num,grade):
    
    patient_path = patient 
    tl.files.exists_or_mkdir(nii_path+grade+'/'+num+'/')
    pMris = [glob(os.path.join(patient_path, 'VSD.Brain' + '*' + mri_type+'.*' +'/'))for mri_type in mri_types]
  #  print(pMris)
    Patient_ims = [glob(os.path.join(pMri[0],  'VSD*'+'.mha'))[0] for pMri in pMris]
   # print(Patient_ims)
    
    Patient_ims = [sitk.ReadImage(im) for im in Patient_ims]   
    
    for l,im in enumerate(Patient_ims): 
        sitk.WriteImage(im, os.path.join(nii_path+grade+'/'+num, mri_types[l]+'.nii.gz'))

for grade in grades:    
    p_folders, patients = make_patients(grade)
    for i in range(len(patients)):
        mha_to_nii(p_folders[i], patients[i], grade)


# In[3]:


mri_types = ['Flair', 'T1', 'T1c', 'T2']
nii_path = 'Training_nii/'
grades = ['HGG', 'LGG']
data_path = 'brainmasks/'



def make_niimg_masks(patient,gpath, gradePath):
    tl.files.exists_or_mkdir(gpath+str(patient))
    
    
    patient_path = gradePath +  str(patient) + '/'
    pMris = [glob(os.path.join(patient_path, mri_type+'.nii.gz'))for mri_type in mri_types]
    print(pMris)
    niimg = [pMri[0] for pMri in pMris]
    
    niimg = [nilearn.image.smooth_img(img, fwhm=6)for img in niimg]
    print(niimg[0].get_data().shape)
   
    brainmask = compute_multi_epi_mask(niimg)
    
    path1 = gpath + str(patient) + '/'
    nib.save(brainmask, os.path.join(path1, 'brainmask.nii.gz'))
    print(patient_path, path1)
    

    
for grade in grades:
    gpath = data_path + grade +'/'
    gradePath = nii_path + grade +'/'
    num_patients = len(os.listdir(gradePath)) 
    num = num_patients
    print(num) 
    p_folders = [os.path.join(gradePath, patient) for patient in os.listdir(gradePath) ] 

    patients = os.listdir(gradePath)
    print(patients)
    
    
    for patient in patients:
    
        
        make_niimg_masks(patient, gpath,  gradePath)
        
    


# In[2]:


#get data for testing

mri_types = ['Flair', 'T1', 'T1c', 'T2']
path = "BRATS2015_Testing/Testing/"
nii_path = 'Testing_nii/'
grades = ['HGG_LGG']


def get_patient_num(p):
    i1 = p.find('pat')
    
    num = p[i1:]
    return num

def make_patients(grade):
    num_patients = len(os.listdir(path+grade)) 
    num = num_patients
    print(num)
    p_folders = [os.path.join(path+grade, patient) for patient in os.listdir(path+grade) ] 
  #  print(p_folders[0])    
    patients = [get_patient_num(p_folders[i])for i in range(len(p_folders))]
  #  print(patients)
    #create patient image path lists for each type of mri
    return p_folders, patients

def mha_to_nii(patient,num,grade):
    
    patient_path = patient 
    tl.files.exists_or_mkdir(nii_path+grade+'/'+num+'/')
    pMris = [glob(os.path.join(patient_path, 'VSD.Brain' + '*' + mri_type+'.*' +'/'))for mri_type in mri_types]
  #  print(pMris)
    Patient_ims = [glob(os.path.join(pMri[0],  'VSD*'+'.mha'))[0] for pMri in pMris]
   # print(Patient_ims)
    
    Patient_ims = [sitk.ReadImage(im) for im in Patient_ims]   
    
    for l,im in enumerate(Patient_ims): 
        sitk.WriteImage(im, os.path.join(nii_path+grade+'/'+num, mri_types[l]+'.nii.gz'))

for grade in grades:    
    p_folders, patients = make_patients(grade)
    for i in range(len(patients)):
        mha_to_nii(p_folders[i], patients[i], grade)


# In[3]:


#make brainmasks for BRATS2015 test set

mri_types = ['Flair', 'T1', 'T1c', 'T2']
nii_path = 'Testing_nii/'
grades = ['HGG_LGG']
data_path = 'brainmasks_test/'



def make_niimg_masks(patient,gpath, gradePath):
    tl.files.exists_or_mkdir(gpath+str(patient))
    
    
    patient_path = gradePath +  str(patient) + '/'
    pMris = [glob(os.path.join(patient_path, mri_type+'.nii.gz'))for mri_type in mri_types]
    print(pMris)
    niimg = [pMri[0] for pMri in pMris]
    
    niimg = [nilearn.image.smooth_img(img, fwhm=6)for img in niimg]
    print(niimg[0].get_data().shape)
   
    brainmask = compute_multi_epi_mask(niimg)
    
    path1 = gpath + str(patient) + '/'
    nib.save(brainmask, os.path.join(path1, 'brainmask.nii.gz'))
    print(patient_path, path1)
    

    
for grade in grades:
    gpath = data_path + grade +'/'
    gradePath = nii_path + grade +'/'
    num_patients = len(os.listdir(gradePath)) 
    num = num_patients
    print(num) 
    p_folders = [os.path.join(gradePath, patient) for patient in os.listdir(gradePath) ] 

    patients = os.listdir(gradePath)
    print(patients)
    
    
    for patient in patients:
    
        
        make_niimg_masks(patient, gpath,  gradePath)
        
    


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages(dataset,m, n): # shows size of the sample    
    
   # indices=np.random.choice(dataset.shape[0], n)
    fig=plt.figure()   
    for i in range(m, m+2*n):        
        a=fig.add_subplot(2,n,i+1-m)
        if i < m + n : d = dataset[:,i,:]
        else: d = dataset[:,:,i-m-n]
        plt.imshow(d)
        
        # a.set_title(chr(labels[indices[i]]+ord('A')))
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        
    plt.show()


# In[9]:


im = nib.load('brainMasks/1/brainmask.nii.gz').get_data()
print (np.min(im), np.max(im), np.sum(im))
print(np.sum(im == 1))

#showDataImages(im,80,19)


# In[6]:


data_path = 'brainmasks/'
path = "ISLES2017_Training/"
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1] 

patients = sorted([int(p_folders[i][28:])for i in range(len(p_folders))])
masks = [glob(os.path.join(data_path + str(patients[l]) + '/', '*'+'.nii.gz'))[0] for l in range(len(patients))]
                                           
def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

for i, m in enumerate(masks):
    im = nib.load(m).get_data()
    shape = im.shape
    print(patients[i], shape, bbox2_3D(im)) 


# In[ ]:




