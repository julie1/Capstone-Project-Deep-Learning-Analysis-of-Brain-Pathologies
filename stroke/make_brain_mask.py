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


mri_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
path = "ISLES2017_Training/"

data_path = 'brainMasks/'

num_patients = len(os.listdir(path)) - 1
num = num_patients
print(num)
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1] 

patients = sorted([int(p_folders[i][28:])for i in range(len(p_folders))])
print(patients)
#create patient image path lists for each type of mri



def make_niimg_masks(i):
    tl.files.exists_or_mkdir(data_path+str(i))
    
    
    patient_path = path + 'training_' + str(i) + '/'
    pMris = [glob(os.path.join(patient_path, '*.Brain.XX.O.MR_' + mri_type+'.*'))for mri_type in mri_types[1:]]
  #  print(pMris)
    niimg = [glob(os.path.join(pMri[0], '*'+'.nii'))[0] for pMri in pMris]
    affines = [nib.load(img).affine for img in niimg]
   # print(nib.load(niimg[0]).affine)
    niimg = [nib.load(img).get_data() for img in niimg]
    print (niimg[0].shape)
    
  #  niimg = [ndimage.zoom(img, (1,1,6)) for img in niimg]
    niimg = [nib.Nifti1Image(niimg[j], affine=affines[j]) for j in range(len(niimg))]
    niimg = [nilearn.image.smooth_img(img, fwhm=6)for img in niimg]
  #  print(niimg[0].get_data().shape)
   # niimg = [nilearn.image.resample_img(img, target_affine=np.eye(4))for img in niimg]
  #  print(niimg.get_affine() )
   # niimg = nilearn.image.mean_img(niimg)
  #  brainmask = [nilearn.masking.compute_epi_mask(img) for img in niimg]
  #  brainmask = [nilearn.image.resample_img(img, target_affine=np.eye(3))for img in brainmask]
    
    brainmask = compute_multi_epi_mask(niimg)
    affine = brainmask.affine
  #  print(brainmask.get_data().shape)
  #  print(brainmask.get_affine() )
    pMris1 = glob(os.path.join(patient_path, '*.Brain.XX.O.MR_' + '4DPWI'+'.*'))    
    im = glob(os.path.join(pMris1[0], '*'+'.nii'))[0]
    im = nib.load(im).get_data()    
    im = np.squeeze(im)
    im = np.mean(im, axis=3)#added after several times without this
  #  im = ndimage.zoom(im, (1,1,6))
    im = nib.Nifti1Image(im, affine=affine)
    niimg1 = nilearn.image.smooth_img(im, fwhm=6)
    
  #  print(niimg1.get_affine())
  #  niimg1 = nilearn.image.mean_img(niimg1)
    
    brainmask1 = compute_epi_mask(niimg1)
  #  print(brainmask1.get_data().shape)
    brainmask2 = nilearn.masking.intersect_masks([brainmask, brainmask1], threshold=.8)
    path1 = data_path + str(i) + '/'
    nib.save(brainmask2, os.path.join(path1, 'brainmask.nii.gz'))
    print(patient_path, path1)
    

    
    
    
    
for i in patients:
    
        
    make_niimg_masks(i)
        
    


# In[ ]:


mri_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
path = 'examples/configFiles/deepMedic/'
data_path = 'examples/dataForStrokes/'

tvt = ['train', 'valid', 'test']


def make_niimg_masks(folder, i):
    if folder == 'valid': path2 = path + 'train/validation/' + str(i) + '/'
    else: path2 = path + folder + '/' + str(i) + '/'
    name = folder + 'ROImasks'
    
    paths = []
    path1 = data_path + folder + '/' + str(i) + '/'
    if folder == 'valid': path_home = '../../../../'
    else: path_home = '../../../'
    
    
    patients = [os.path.join(path1, patient) for patient in os.listdir(path1)] 
  #  print (patients)

    new_file = open(path2+name+'.cfg', 'w')
    new_file.close()  
    
    for patient in patients: 
        
        niimg = [glob(os.path.join(patient, mri_type+'.nii.gz')) for mri_type in mri_types]
        niimg = nilearn.image.smooth_img(niimg, fwhm=6)
        nilearn.image.mean_img(niimg)
        paths.append(niimg)
        brainmask = compute_multi_background_mask(niimg)
        nib.save(brainmask, os.path.join(patient, 'brainmask.nii.gz'))
        with open(path2+name+'.cfg', 'a') as f:            
            f.write(path_home+patient[9:]+'{}'.format('/brainmask.nii.gz')) 
            f.write('\n')
        
        print (path1)
        print (patient)
    return paths      
    
for i in range(5):
    for folder in tvt:
        
        make_niimg_masks(folder, i)
        
    


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


# In[4]:


im = nib.load('brainMasks/1/brainmask.nii.gz').get_data()
print (np.min(im), np.max(im), np.sum(im))
print(np.sum(im == 1))
im1 = nib.load('brainMasks/1/brainmask.nii.gz')
print (im1.affine)
im2 = ndimage.zoom(im, (1,1,6))
print(im2.shape)
im = nib.Nifti1Image(im, affine=im1.affine)
#showDataImages(im,80,19)


# In[6]:


data_path = 'brainMasks/'
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




