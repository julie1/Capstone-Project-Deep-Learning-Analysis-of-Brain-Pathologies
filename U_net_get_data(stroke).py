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
import math


# In[2]:



statinfo = os.stat('ISLES2017_Testing.zip')
print (statinfo.st_size)
statinfo = os.stat('ISLES2017_Training.zip')
print (statinfo.st_size)
                


# In[3]:


def maybe_extract(path, filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
  newpath = root
  if os.path.isdir(newpath) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:      
    os.makedirs(newpath)
    print('Extracting data for %s. This may take a while. Please wait.' % newpath)
    zip_ref = zp(path+filename)    
    zip_ref.extractall(newpath)
    zip_ref.close()
  
  data_files = [
    os.path.join(newpath, f) for f in os.listdir(newpath) 
   ]
  
  
  return data_files

path = '/home/julie/U-net_code_stroke/'

ISLES2017_Testing = maybe_extract(path, 'ISLES2017_Testing.zip')
ISLES2017_Training = maybe_extract(path, 'ISLES2017_Training.zip')
print (len(ISLES2017_Testing), len(ISLES2017_Training))
print (ISLES2017_Testing[0])
print (ISLES2017_Training[0])


# In[2]:


path_to_image = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_1/VSD.Brain.XX.O.MR_4DPWI.127015/VSD.Brain.XX.O.MR_4DPWI.127015.nii"
image = nib.load(path_to_image).get_data()
print(image.shape)


# In[2]:


path_to_image = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_43/SMIR.Brain.XX.O.MR_4DPWI.188919/SMIR.Brain.XX.O.MR_4DPWI.188919.nii"
image = nib.load(path_to_image).get_data()
print(image.shape)


# In[4]:



print(image.dtype )
print(np.max(image))
print(np.min(image))
print(np.std(image))

print(np.mean(image))

path2 = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_1/VSD.Brain.XX.O.MR_ADC.128020/VSD.Brain.XX.O.MR_ADC.128020.nii"
im2 = nib.load(path2).get_data()
print(im2.shape)
print(im2.dtype )
print(np.max(im2))
print(np.min(im2))
print(np.std(im2))

path3 = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_1/VSD.Brain.XX.O.OT.128050/VSD.Brain.XX.O.OT.128050.nii"
im3 = nib.load(path3).get_data()
print(im3.shape)
print(im3.dtype )
print(np.max(im3))
print(np.min(im3))
print(np.std(im3))
print(np.sum(im3))
path4 = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_10/VSD.Brain.XX.O.MR_4DPWI.127023/VSD.Brain.XX.O.MR_4DPWI.127023.nii"
im4 = nib.load(path4).get_data()
print(im4.shape)
print(im4.dtype )
print(np.max(im4))
print(np.min(im4))
print(np.std(im4))
print(np.sum(im4))
path5 = "/home/julie/U-net_code_stroke/ISLES2017_Training/training_14/VSD.Brain.XX.O.MR_4DPWI.127055/VSD.Brain.XX.O.MR_4DPWI.127055.nii"
im5 = nib.load(path5).get_data()
print(im5.shape)
print(im5.dtype )
print(np.max(im5))
print(np.min(im5))
print(np.std(im5))
print(np.sum(im5))


# In[4]:


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


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages5(dataset,m, n, i, j): # shows size of the sample    
    
   # indices=np.random.choice(dataset.shape[0], n)
    fig=plt.figure()   
    for k in range(m, m+2*n):
        a=fig.add_subplot(2,n,k+1-m)
        if k < m + n : d = dataset[i,:,:,0,k]
        else: d = dataset[:,:,j,0,k]
        plt.imshow(d)
        
        # a.set_title(chr(labels[indices[i]]+ord('A')))
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        
    plt.show()


# In[14]:


showDataImages5(np.expand_dims(image,3),20,5,80,10)
showDataImages(im2,80,19)
showDataImages(im3,70,19)
showDataImages5(im4,20,5,80,10)
showDataImages5(im5,20,5,80,10)


# In[2]:


path = "ISLES2017_Training/"

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']

num_patients = len(os.listdir(path)) - 1
num = num_patients
print(num)

#create patient image path lists for each type of mri
def create_patients(path, num):
    
    mris = []
    
    p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1] 
    
    
    p_folders = sorted(p_folders, key = lambda path: getNum(path))
   # for f in p_folders: print(f,getNum(f))
  #  print(p_folders)
    for mri_type in data_types:
        mritype = []        
        for i, p in enumerate(p_folders):            
            if mri_type != 'OT': patient = sorted(glob(os.path.join(p, '*.Brain.XX.O.MR_' + mri_type+'.*')))  
            else: patient = sorted(glob(os.path.join(p, '*.Brain.XX.O.' + mri_type+'.*'))) 
           # print (patient)
            mri_name = glob(os.path.join(patient[0], '*'+'.nii'))[0]
            mritype.append(mri_name)
        mris.append(mritype)
        #mri_data, _ = load(mri)
            
            
        
       
                
    return mris

def getNum(patient):
    
    a = patient.find("training_")
    
    
    return int(patient[a+9:])

mris = create_patients(path, num)
#print (mris[0])
mri_data = {}
for i, d in enumerate(data_types): 
    mri_data[d] = {}
    for mri in mris[i]:
        im = nib.load(mri).get_data()
            
        
        shape = im.shape
        mri_data[d][mri] = shape 
        
                
        del im

with open('mri_data.json', 'w') as f:
    json.dump(mri_data, f)        
    


# In[8]:


os.listdir('.')


# In[3]:


path = "ISLES2017_Training/"
path1 = "brainMasks/"
#image_size = (240, 240, 30)
target_affine = np.diag((1, 1, 1)) #np.diag((1, 1, 1))#np.diag((1, 1, 5))
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1]
patients = sorted([int(p_folders[i][28:])for i in range(len(p_folders))])



num_patients = len(os.listdir(path)) - 1
num = num_patients
print(num)

mask_names = [glob(os.path.join(path1 + str(patients[k]) + '/', 'brainmask.nii.gz'))[0] for k in range(len(patients))]


with open('mri_data.json') as f:
    mri_data = json.load(f)
from nilearn.image import resample_img
def resample(im):
  #  print(im.shape)
    
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
  #  im = nib.load(im).get_data()
  #  nib.save(im, os.path.join('.','im.nii'))
  #  im = nib.load('im.nii')
    
   # print(im.shape)
    return im.get_data() 


masks = [resample(nib.load(mask_name).get_data()).astype(np.int16) for mask_name in  mask_names]
S = np.sum([np.sum(mask) for mask in masks])
print(S)


def get_mean(mris):
    mean = {}
    for i, d in enumerate(data_types):
        sum1 = 0      
        
        
        
        for j, mri in enumerate(mris[i]):
            
            mask = masks[j]
            im = nib.load(mri).get_data()
            im = np.squeeze(im)
            
           # print(im.shape)
            if i == 0:
                im = np.mean(im, axis=3)
                
                
                #instead of this do mean first
             #   ims = [im[:,:,:, t] for t in range(im.shape[3])]
             #   ims = [resample(img) for img in ims]
           #     print(ims[0].shape)
             #   im = np.stack(ims, axis=3)
             #   print(im.shape)
             #   im = np.mean(im, axis=3) 
           # im = ndimage.zoom(im, (1,1,6))
            im = resample(im)
         #   print(mask.shape, im.shape)
            sum1 += np.sum(im*mask)
            del im
        mean[d] = float(sum1) / S            
            
        
    print (mean)   
    return mean

def get_mean_std(mris):
    mean_std = {}
    for i, d in enumerate(data_types):
        sum2 = 0      
        mean = get_mean(mris)[d]
        
        
        for j, mri in enumerate(mris[i]):
            
            mask = masks[j]
            im = nib.load(mri).get_data()
            im = np.squeeze(im)
            
            if i == 0: 
                ims = [im[:,:,:, t] for t in range(im.shape[3])]
               # ims = [resample(img) for img in ims]
                im = np.stack(ims, axis=3)
                im = np.mean(im, axis=3)
          #  im = ndimage.zoom(im, (1,1,6))
            im = resample(im)
            sum2 += np.sum((im - mean)**2 * mask)
            del im
        mean_std[d] = (mean, math.sqrt(float(sum2) / S ))           
        print (d, mean_std[d])    
        
       
    return mean_std




mris = create_patients(path, num)

print (mris[0][0])
mean_std = get_mean_std(mris)
#for key in mean_std: print (type(mean_std[key][0]))
print(mean_std['4DPWI'])

with open('mean_std.json', 'w') as f1:
    json.dump(mean_std, f1)


    


# In[3]:


path = "ISLES2017_Training/"
path1 = "brainMasks/"

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1]
p_folders = sorted(p_folders, key = lambda path: getNum(path))
patients = [int(p_folders[i][28:])for i in range(len(p_folders))]

def get_roi_mean(masks, mris, d):
    sum1 = 0        
        
    for j, mri in enumerate(mris):

        mask = masks[j]
        im = nib.load(mri).get_data() 
      
        sum1 += np.sum(im*mask)
        
    mean = float(sum1) / S            
            
        
    print (d, mean)   
    return mean

def get_roi_std(masks, mris, d):
   
    sum2 = 0      
    mean = get_roi_mean(masks, mris, d)


    for j, mri in enumerate(mris):

        mask = masks[j]
        im = nib.load(mri).get_data() 
        sum2 += np.sum((im - mean)**2 * mask)
        
    mean_std = (mean, math.sqrt(float(sum2) / S ))           
    print (d, mean, mean_std)    

       
    return mean_std

def getNum(patient):
    
    a = patient.find("training_")
    
    
    return int(patient[a+9:])
 


# In[4]:


#normalize mri images and move to patientIms

path = "ISLES2017_Training/"

#image_size = (240, 240, 30)
target_affine = np.diag((1, 1, 1))#np.diag((1, 1, 1))#np.diag((1, 1, 5))
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1]
p_folders = sorted(p_folders, key = lambda patient: getNum(patient))
patients = [int(p_folders[i][28:])for i in range(len(p_folders))]
mris = create_patients(path, num)
data_path = 'patientIms/'
mask_names = [glob(os.path.join(path1 + str(patients[k]) + '/', 'brainmask.nii.gz'))[0] for k in range(len(patients))]


with open('mean_std.json') as f:
    mean_std = json.load(f)

with open('mri_data.json') as f1:
    mri_data = json.load(f1)

from nilearn.image import resample_img
def resample(im):
  #  print(im.shape)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
  #  im = nib.load(im).get_data()
  #  nib.save(im, os.path.join('.','im.nii'))
  #  im = nib.load('im.nii')
    
   # print(im.shape)
    return im.get_data()

masks = [resample(nib.load(mask_name).get_data()).astype(np.int16) for mask_name in  mask_names]
S = np.sum([np.sum(mask) for mask in masks])
print(S)

def normalize_mri(mri, d):
    im = nib.load(mri).get_data()            
    im = np.squeeze(im)
    im = nib.Nifti1Image(im, affine=np.eye(4))
  #  im = nilearn.image.resample_img(im, target_affine=target_affine)
    im = im.get_data()
    shape = im.shape
    
    mean, std_dev = mean_std[d]   
      
    if d == '4DPWI':
            
        im = np.mean(im, axis=3)
        #     print (im.shape)
  #  im = ndimage.zoom(im, (1,1,6))
    im = resample(im)   
    #im = nilearn.image.resample_img(im, target_affine=target_affine)    
    if d != 'OT': im = (im - mean) / std_dev 
    else: im = im.astype(np.int16)
    #print(imt1.shape)
    return im

    
    
    
#subtract the mean and divide by the standard deviation of patients for 7 MRI types(stroke)
def normalize_patients(d, mean_std):
    for l, mri in enumerate(mris[data_types.index(d)]):
        im = normalize_mri(mri, d)
        im1 = nib.Nifti1Image(im, affine=np.eye(4))
        tl.files.exists_or_mkdir(data_path+str(patients[l]))        
        path1 = data_path+str(patients[l])+'/'        
        nib.save(im1, os.path.join(path1, d +'.nii.gz'))
    mripaths = [data_path+str(patients[i])+'/' for i in range(len(patients))]
    mrisd = [os.path.join(mripaths[i], d +'.nii.gz') for i in range(len(patients))]
    
                              
    print (get_roi_std(masks, mrisd, d))
           
    

for d in data_types:
    normalize_patients(d, mean_std)
           






# In[5]:


#resample and move masks into patientIms 

from nilearn.image import resample_img
def resample(im):
  #  print(im.shape)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
  #  im = nib.load(im).get_data()
  #  nib.save(im, os.path.join('.','im.nii'))
  #  im = nib.load('im.nii')
    
   # print(im.shape)
    return im.get_data()



path = "ISLES2017_Training/"
path1 = "brainMasks/"
target_affine = np.diag((1, 1, 1))
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1]
p_folders = sorted(p_folders, key = lambda patient: getNum(patient))
patients = [int(p_folders[i][28:])for i in range(len(p_folders))]
mris = create_patients(path, num)
data_path = 'patientIms/'
mask_names = [glob(os.path.join(path1 + str(patients[k]) + '/', 'brainmask.nii.gz'))[0] for k in range(len(patients))]
masks = [resample(nib.load(mask_name).get_data()).astype(np.int16) for mask_name in  mask_names]


for i, mask in enumerate(masks):
    path2 = data_path+str(patients[i])+'/'
    im = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.save(im, os.path.join(path2, 'brainmask.nii.gz'))


# In[6]:


#make sure ground is 0's and 1's
d = 'OT'
im = nib.load(os.path.join('patientIms/'+'1', d +'.nii.gz')).get_data()
im1 = nib.load(os.path.join('patientIms/'+'1', d +'.nii.gz'))
print (im1.affine)
print (np.max(im), np.min(im), np.sum(im), im.shape)


# In[3]:


#compare to normalized, unmagnified image
path = '/home/julie/deepmedic/examples/dataForStrokes/test/patient_1/'
imOrig = nib.load(os.path.join(path, d +'.nii.gz')).get_data()
print (np.max(imOrig), np.min(imOrig), np.sum(imOrig), imOrig.shape)
imOrig1 = ndimage.zoom(imOrig, (1,1,6))
print (np.max(imOrig1), np.min(imOrig1), np.sum(imOrig1), imOrig1.shape)


# In[5]:


path = "ISLES2017_Training/"

def getNum(patient):
    
    a = patient.find("training_")
    
    
    return int(patient[a+9:])
p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1]
p_folders = sorted(p_folders, key = lambda patient: getNum(patient))
patients = [int(p_folders[i][28:])for i in range(len(p_folders))]
print (patients)
d = 'OT'
m = 'brainmask'
maskSum = {}
for j in range(len(patients)):
    im = nib.load(os.path.join('patientIms/'+str(patients[j]), d +'.nii.gz')).get_data()
    mask = nib.load(os.path.join('patientIms/'+str(patients[j]), m +'.nii.gz')).get_data()
    im1 = im * mask
    print (patients[j], ':', np.max(im), np.min(im), np.sum(im), im.shape)
    print (patients[j], ':', np.max(im1), np.min(im1), np.sum(im1), im1.shape)
    maskSum[str(patients[j])] = int(np.sum(im1))
with open('maskSum.json', 'w') as f1:
    json.dump(maskSum, f1)


# In[5]:


image_size = (240, 240, 30)
target_affine = np.diag((1, 1, 5))
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
with open('mean_std.json') as f:
    mean_std = json.load(f)  
    
with open('mri_data.json') as f1:
    mri_data = json.load(f1)
    



def crop_pad_normalize_mri(mri, d):
    im = nib.load(mri).get_data()            
    im = np.squeeze(im)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
    im = im.get_data()
    shape = im.shape
    
    mean, std_dev = mean_std[d]         
   # print(shape)
   
           
    if shape[:3] != image_size:
        pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape)]
       # print( pads)
     #   print(mri, shape, shape1, pads)
        cuts = [((sz[0] - abs(sh[0]))/2 , (sz[1] - abs(sh[1]))/2 )  for sz,sh in zip(pads, pads)]

      #  print (d, cuts, shape)
        im = im[-cuts[0][0]:shape[0]+cuts[0][1],-cuts[1][0]:shape[1]+cuts[1][1],
                -cuts[2][0]:shape[2]+cuts[2][1]]

        shape1 = im.shape
      #  print (shape1)
        pads1 = pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape1)]
      #  print (pads1)
        if d == '4DPWI':
            if pads1 != 0: im = np.lib.pad(im, (pads1[0],pads1[1],pads1[2],(0,0)),'constant')
            im = np.mean(im, axis=3)            
       #     print (im.shape)

        elif d != '4DPWI' and pads1 != 0:
        #   print(i, d, shape1, pads1)
            im =  np.lib.pad(im[:,:,:], (pads1[0],pads1[1],pads1[2]),'constant')
            
    
    
    assert im.shape == image_size
    if d != 'OT': im = (im - mean) / std_dev 
    #print(imt1.shape)
    return im

    
    
    
#subtract the mean and divide by the standard deviation of patients for 7 MRI types(stroke)
def normalize_patients(d, mean_std):
    mris = mri_data[d].keys()
    num = len(mris)
    print(num)
    mean, std_dev = mean_std[d]
    shape = tuple([num]) + tuple(image_size)
    
    dtype = nib.load(mris[0]).get_data().dtype
    mris_norm = np.ndarray(shape=shape, dtype=dtype)
    
    mris_norm[:] = [crop_pad_normalize_mri(mri, d) for mri in mris]
    mris_norm = mris_norm[:num, ...] 
    
            
    mris_norm = mris_norm[:num, ...] 
    
    
    
    epsilon=1e-7
    print (d, 'mean:', mean, 'std-dev:', std_dev)
    print (d, 'mean:', float(np.mean(mris_norm)), 'std-dev:', float(np.std(mris_norm)))
   
    return mris_norm








for i, _ in enumerate(data_types):
    
    mri_type = data_types[i]
    
    mris_norm = normalize_patients(mri_type, mean_std)
    
    
    with h5py.File('mris_norm_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("mris_norm_"+ mri_type,  data=mris_norm)
        del mris_norm


# In[3]:


image_size = (240, 240, 30)
target_affine = np.diag((1, 1, 1))#np.diag((1, 1, 5))
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
with open('mean_std.json') as f:
    mean_std = json.load(f)

with open('mri_data.json') as f1:
    mri_data = json.load(f1)
path = "ISLES2017_Training/"

from nilearn.image import resample_img
def resample(im):
  #  print(im.shape)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
  #  im = nib.load(im).get_data()
  #  nib.save(im, os.path.join('.','im.nii'))
  #  im = nib.load('im.nii')
    
   # print(im.shape)
    return im.get_data()

def crop_pad_normalize_mri(mri, d):
    im = nib.load(mri).get_data()            
    im = np.squeeze(im)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
    im = im.get_data()
    shape = im.shape
    
    mean, std_dev = mean_std[d]         
   # print(shape)
   
           
    if shape[:3] != image_size:
        pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape)]
       # print( pads)
     #   print(mri, shape, shape1, pads)
        cuts = [((sz[0] - abs(sh[0]))/2 , (sz[1] - abs(sh[1]))/2 )  for sz,sh in zip(pads, pads)]

      #  print (d, cuts, shape)
        im = im[-cuts[0][0]:shape[0]+cuts[0][1],-cuts[1][0]:shape[1]+cuts[1][1],
                -cuts[2][0]:shape[2]+cuts[2][1]]

        shape1 = im.shape
      #  print (shape1)
        pads1 = pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape1)]
      #  print (pads1)
        if d == '4DPWI':
            if pads1 != 0: im = np.lib.pad(im, (pads1[0],pads1[1],pads1[2],(0,0)),'constant')
            im = np.mean(im, axis=3)            
       #     print (im.shape)

        elif d != '4DPWI' and pads1 != 0:
        #   print(i, d, shape1, pads1)
            im =  np.lib.pad(im[:,:,:], (pads1[0],pads1[1],pads1[2]),'constant')
            
    
    
    assert im.shape == image_size
    if d != 'OT': im = (im - mean) / std_dev 
    #print(imt1.shape)
    return im

    
    
    
#subtract the mean and divide by the standard deviation of patients for 7 MRI types(stroke)
def normalize_patients(d, mean_std):
    mris = mri_data[d].keys()
    num = len(mris)
    print(num)
    mean, std_dev = mean_std[d]
    shape = tuple([num]) + tuple(image_size)
    
    dtype = nib.load(mris[0]).get_data().dtype
    mris_norm = np.ndarray(shape=shape, dtype=dtype)
    
    mris_norm[:] = [crop_pad_normalize_mri(mri, d) for mri in mris]
    mris_norm = mris_norm[:num, ...] 
    
            
    mris_norm = mris_norm[:num, ...] 
    
    
    
    epsilon=1e-7
    print (d, 'mean:', mean, 'std-dev:', std_dev)
    print (d, 'mean:', float(np.mean(mris_norm)), 'std-dev:', float(np.std(mris_norm)))
   
    return mris_norm








for i, _ in enumerate(data_types):
    
    mri_type = data_types[i]
    
    mris_norm = normalize_patients(mri_type, mean_std)
    
    
    with h5py.File('mris_norm_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("mris_norm_"+ mri_type,  data=mris_norm)
        del mris_norm


# In[3]:


mris = create_patients(path, num)
for i, d in enumerate(data_types):
    im = nib.load(mris[i][0]).get_data()
    print (im.dtype)
        
        
        
            
            


# In[9]:


data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']

with open('mri_data.json') as f1:
    mri_data = json.load(f1)
print (mri_data.keys())

for d in data_types:
    print(d)
    for mri in mri_data[d].keys(): print (mri[28:30], mri_data[d][mri])
        
mris = []


# In[25]:


img = "ISLES2017_Training/training_43/SMIR.Brain.XX.O.MR_4DPWI.188919/SMIR.Brain.XX.O.MR_4DPWI.188919.nii"
source_affine = np.eye(4)
image = nib.load(img).get_data()
image = np.squeeze(image)[:, :, :, 40]
print(image.shape)
img = nib.Nifti1Image(image, affine=source_affine)
from nilearn.image import resample_img

img_in_mm_space = resample_img(img, target_affine=np.eye(4),
                           target_shape=(240, 240, 30))
target_affine_3x3 = np.eye(3) * 3
target_affine_4x4 = np.eye(4) * 3
target_affine_4x4[3, 3] = 1.
img_3d_affine = resample_img(img, target_affine=target_affine_3x3)
img_4d_affine = resample_img(img, target_affine=target_affine_4x4)
target_affine_mm_space_offset_changed = np.eye(4)
target_affine_mm_space_offset_changed[:3, 3] =     img_3d_affine.affine[:3, 3]
vmax = image.max()
img_3d_affine_in_mm_space = resample_img(
    img_3d_affine,
    target_affine=target_affine_mm_space_offset_changed,
    target_shape=(np.array(img_3d_affine.shape) * 2).astype(int))

img_4d_affine_in_mm_space = resample_img(
    img_4d_affine,
    target_affine=np.eye(4),
    target_shape=(np.array(img_4d_affine.shape) * 2).astype(int))


plt.figure()
plt.imshow(image[:,:,15], interpolation="nearest", vmin=0, vmax=vmax)
plt.title("The original data in voxel space")

plt.figure()
plt.imshow(img_in_mm_space.get_data()[:,:,15], vmin=0, vmax=vmax)
plt.title("The original data in mm space")

plt.figure()
plt.imshow(img_3d_affine_in_mm_space.get_data()[:,:,7],
           vmin=0, vmax=vmax)
plt.title("Transformed using a 3x3 affine -\n leads to "
          "re-estimation of bounding box")

plt.figure()
plt.imshow(img_4d_affine_in_mm_space.get_data()[:,:,7],
           vmin=0, vmax=vmax)
plt.title("Transformed using a 4x4 affine -\n Uses affine anchor "
          "and estimates bounding box size")

plt.show()


# In[2]:


image_size = (240, 240, 30)
target_affine = np.diag((1, 1, 5))
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
with open('mean_std.json') as f:
    mean_std = json.load(f)

with open('mri_data.json') as f1:
    mri_data = json.load(f1)


def crop_pad_normalize_mri(mri, d):
    im = nib.load(mri).get_data()            
    im = np.squeeze(im)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine, target_shape=image_size)
    im = im.load(im).get_data()
    shape = im.shape
    
    mean, std_dev = mean_std[d]         
   # print(shape)
   
           
    if shape[:3] != image_size:
        pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape)]
       # print( pads)
     #   print(mri, shape, shape1, pads)
        cuts = [((sz[0] - abs(sh[0]))/2 , (sz[1] - abs(sh[1]))/2 )  for sz,sh in zip(pads, pads)]

      #  print (d, cuts, shape)
        im = im[-cuts[0][0]:shape[0]+cuts[0][1],-cuts[1][0]:shape[1]+cuts[1][1],
                -cuts[2][0]:shape[2]+cuts[2][1]]

        shape1 = im.shape
      #  print (shape1)
        pads1 = pads = [((sz - sh)/2 , (sz - sh)/2 + (sz -sh)%2) for sz,sh in zip(image_size, shape1)]
      #  print (pads1)
        if d == '4DPWI':
            if pads1 != 0: im = np.lib.pad(im, (pads1[0],pads1[1],pads1[2],(0,0)),'constant')
            im = np.mean(im, axis=3)            
       #     print (im.shape)

        elif d != '4DPWI' and pads1 != 0:
        #   print(i, d, shape1, pads1)
            im =  np.lib.pad(im[:,:,:], (pads1[0],pads1[1],pads1[2]),'constant')
            
    
    
    assert im.shape == image_size
    if d != 'OT': im = (im - mean) / std_dev 
    #print(imt1.shape)
    return im

    
    
    
#subtract the mean and divide by the standard deviation of patients for 7 MRI types(stroke)
def normalize_patients(d, mean_std):
    mris = mri_data[d].keys()
    num = len(mris)
    print(num)
    mean, std_dev = mean_std[d]
    shape = tuple([num]) + tuple(image_size)
    
    dtype = nib.load(mris[0]).get_data().dtype
    mris_norm = np.ndarray(shape=shape, dtype=dtype)
    
    mris_norm[:] = [crop_pad_normalize_mri(mri, d) for mri in mris]
    mris_norm = mris_norm[:num, ...] 
    
            
    mris_norm = mris_norm[:num, ...] 
    
    
    
    epsilon=1e-7
    print (d, 'mean:', mean, 'std-dev:', std_dev)
    print (d, 'mean:', float(np.mean(mris_norm)), 'std-dev:', float(np.std(mris_norm)))
   
    return mris_norm








for i, _ in enumerate(data_types):
    
    mri_type = data_types[i]
    
    mris_norm = normalize_patients(mri_type, mean_std)
    
    
    with h5py.File('mris_norm_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("mris_norm_"+ mri_type,  data=mris_norm)
        del mris_norm


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
def showDataImages(dataset,n): # shows size of the sample    
    
   # indices=np.random.choice(dataset.shape[0], n)
    fig=plt.figure()   
    for i in range(n):
        a=fig.add_subplot(1,n,i+1)
        d = dataset[:, :, 10+i]
        
        plt.imshow(d)
        
        # a.set_title(chr(labels[indices[i]]+ord('A')))
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        
    plt.show()


# In[5]:


with open('mri_data.json') as f1:
    mri_data = json.load(f1)

data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
for i, d in enumerate(data_types):
    
    mri_type = data_types[i]
    
   
    
    with h5py.File('mris_norm_' + mri_type + '.h5', 'r') as hf:
        mris_norm = hf['mris_norm_' + mri_type ][:]
        
    showDataImages(mris_norm[20],3) 
    del mris_norm
    


# In[3]:


with open('mri_data.json') as f1:
    mri_data = json.load(f1)
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']

for i, d in enumerate(data_types1):
    
    mri_type = data_types1[i]
    
   
    
    with h5py.File('norm_mris_' + mri_type + '.h5', 'r') as hf:
        norm_mris = hf['norm_mris_' + mri_type ][:]
        
    print (d, norm_mris.shape, norm_mris.dtype, np.min(norm_mris), np.max(norm_mris), np.mean(norm_mris))
        
    showDataImages(norm_mris[10],80,5) 
    del norm_mris
    
mri_type = 'OT'
    
   
    
with h5py.File('mris_ground_' + mri_type + '.h5', 'r') as hf:
    mris_ground = hf['mris_ground_' + mri_type ][:]
    print (np.sum(mris_ground[0]))
    print('OT', mris_ground.shape, mris_ground.dtype, np.min(mris_ground), np.max(mris_ground), np.mean(mris_ground))
    showDataImages(mris_ground[10],100,5) 
    del mris_ground


# In[4]:


with open('mri_data.json') as f1:
    mri_data = json.load(f1)

print (mri_data['4DPWI'].values())


# In[2]:


#create repetitive ground for 4DPWI
image_size = [200, 200, 24]

with open('mri_data.json') as f1:
    mri_data = json.load(f1)
    
with h5py.File('mris_ground_OT.h5', 'r') as hf:
    mris_ground = hf['mris_ground_OT'][:]

times_4DPWI = [mri_data['4DPWI'].values()[i][0][3] for i in range(len(mri_data['4DPWI'].keys()))]
shape = tuple([sum(times_4DPWI)]) + tuple(image_size)
mris_ground_repeat = np.ndarray(shape=shape, dtype=np.uint8)
n = 0
for i, r in enumerate(times_4DPWI):
    
    mris_ground_repeat[n:n+r, ...] = mris_ground[i]
    n = n+r
mris_ground_repeat = mris_ground_repeat[:n, ...]
    
del mris_ground
print (mris_ground_repeat.shape)

with h5py.File('mris_ground_repeat.h5', 'w') as hf:
    hf.create_dataset("mris_ground_repeat",  data=mris_ground_repeat)
    del mris_ground_repeat

    


# In[3]:


#split into train, test, valid:patients 1-25 train, 26-34 valid, 35-43 test

with open('mri_data.json') as f1:
    mri_data = json.load(f1)
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
times = [mri_data['4DPWI'].values()[i][0][3] for i in range(len(mri_data['4DPWI'].keys()))]

for i, d in enumerate(data_types1):
    
    mri_type = data_types1[i]
    if i == 0:
        train = sum(times[:25])
        valid = train + sum(times[25:34])
        test = valid + sum(times[34:43])
        print (train, valid, test)
    else: train, valid, test = 25, 34, 43
        
   
    
    with h5py.File('norm_mris_' + mri_type + '.h5', 'r') as hf:
        norm_mris_train = hf['norm_mris_' + mri_type ][:train]
        norm_mris_valid = hf['norm_mris_' + mri_type ][train:valid]
        norm_mris_test = hf['norm_mris_' + mri_type ][valid:test]
        del hf
        
    print (d, norm_mris_train.shape, norm_mris_valid.shape, norm_mris_test.shape)
    with h5py.File('norm_mris_train_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("norm_mris_train_"+ mri_type,  data=norm_mris_train)
        del norm_mris_train
    with h5py.File('norm_mris_valid_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("norm_mris_valid_"+ mri_type,  data=norm_mris_valid)
        del norm_mris_valid
    with h5py.File('norm_mris_test_' + mri_type + '.h5', 'w') as hf:
        hf.create_dataset("norm_mris_test_"+ mri_type,  data=norm_mris_test)
        del norm_mris_test
        
    


# In[4]:


path = "/home/julie/U-net_code_stroke/ISLES2017_Training/"

data_types = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax', 'OT']
data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']

num_patients = len(os.listdir(path)) - 1
num = num_patients

with open('mri_data.json') as f1:
    mri_data = json.load(f1)

times = [mri_data['4DPWI'].values()[i][0][3] for i in range(len(mri_data['4DPWI'].keys()))]

#create patient image path lists for each type of mri
def create_patients(path, num):
    
    mris = []
    
    p_folders = [os.path.join(path, patient) for patient in os.listdir(path) if patient.find("_MACOSX") == -1] 
    
    
    p_folders = sorted(p_folders, key = lambda path: getNum(path))
   # for f in p_folders: print(f,getNum(f))
    
    for mri_type in data_types:
        mritype = []        
        for i, p in enumerate(p_folders):            
            if mri_type != 'OT': patient = sorted(glob(os.path.join(p, '*.Brain.XX.O.MR_' + mri_type+'.*')))  
            else: patient = sorted(glob(os.path.join(p, '*.Brain.XX.O.' + mri_type+'.*'))) 
           # print (patient)
            mri_name = glob(os.path.join(patient[0], '*'+'.nii'))[0]
            mritype.append(mri_name)
        mris.append(mritype)
        #mri_data, _ = load(mri)
            
            
        
       
                
    return mris

def getNum(patient):
    
    a = patient.find("training_")
    
    
    return int(patient[a+9:])


# create combined array of various MRI types by reducing one dimension and repeating 6 MRIS along time axis of
#4DPWI

mris_train_paths = create_patients(path, num)[:25]
mris_valid_paths = create_patients(path, num)[25:34]
mris_test_paths = create_patients(path, num)[34:43]

with h5py.File('mris_paths.h5', 'w') as hf:
    hf.create_dataset("mris_train_paths",  data=mris_train_paths)
    hf.create_dataset("mris_valid_paths",  data=mris_valid_paths)
    hf.create_dataset("mris_test_paths",  data=mris_test_paths)
    del hf


# In[2]:


data_types1 = ['4DPWI', 'ADC', 'rCBV', 'rCBF', 'MTT', 'TTP', 'Tmax']
with open('mri_data.json') as f1:
    mri_data = json.load(f1)

times = [mri_data['4DPWI'].values()[i][0][3] for i in range(len(mri_data['4DPWI'].keys()))]



#create stack for each patient
def create_mriStacks(name, i):
    
    stack = []
    
  #  c = 0
    c =  sum(times[:i])
        
    for m in range(times[i]):
            
        data = []
            
            
            
        for j, d in enumerate(data_types1):
            if j == 0:
                with h5py.File(name+d+'.h5', 'r') as hf:
                    data.append(np.reshape(hf[name+d][c:c+1,...], (200,200,24)))    
                    del hf
            else:
                
                with h5py.File(name+d+'.h5', 'r') as hf:
                    data.append(np.reshape(hf[name+d][i:i+1,...], (200,200,24)))    
                    del hf
                           
        
        for l in range(data[0].shape[2]):
            
            stack.append(np.stack((data[k][:,:,l]for k in range(7)), axis=2))
              
                
        c += 1
              
    stack = np.asarray(stack, dtype=np.float32)
    print('stack_patient'+str(i), stack.shape)
    print (stack.nbytes/1e9 )                     
    return stack

for i in range(len(times)):
    with h5py.File('stack_patient'+ str(i + 1) + '.h5', 'w') as hf:
    
        hf.create_dataset("stack_patient" + str(i + 1),  data=create_mriStacks('norm_mris_', i))
        del hf


# In[2]:


stack_shapes = {}

for i in range(43):
    with h5py.File('stack_patient'+ str(i + 1) + '.h5', 'r') as hf:
        stack_patient = hf['stack_patient'+ str(i + 1)][:]
        stack_shapes[i+1] = stack_patient.shape
        print(stack_shapes[i+1])
        del stack_patient, hf
        
with open('stack_shapes.json', 'w') as f:
    json.dump(stack_shapes, f)   


# In[2]:


stack_shapes = {}

for i in range(43):
    with h5py.File('stack_patient'+ str(i + 1) + '.h5', 'r') as hf:
        stack_patient = hf['stack_patient'+ str(i + 1)][:]
        stack_shapes[i+1] = stack_patient.shape
        print(stack_shapes[i+1])
        del stack_patient, hf

with open('stack_shapes.pkl', 'wb') as outfile1:
    pickle.dump(stack_shapes, outfile1, pickle.HIGHEST_PROTOCOL)


# In[8]:


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


# In[7]:


path_to_image = "ISLES2017_Training/training_43/SMIR.Brain.XX.O.MR_4DPWI.188919/SMIR.Brain.XX.O.MR_4DPWI.188919.nii"
path_to_image1 = "ISLES2017_Training/training_1/VSD.Brain.XX.O.MR_4DPWI.127015/VSD.Brain.XX.O.MR_4DPWI.127015.nii"
image = nib.load(path_to_image1)
print(image.header)


# In[33]:



target_affine = np.diag((1,1,6))
im='ISLES2017_Training/training_1/VSD.Brain.XX.O.MR_4DPWI.127015/VSD.Brain.XX.O.MR_4DPWI.127015.nii'
im = nib.load(mris[1][0])
im2 = np.squeeze(im.get_data())
print(im2.shape)
print(im.affine)
im1 = im
print(im1.shape)
def resample(im):
  #  print(im.shape)
    im = nib.Nifti1Image(im, affine=np.eye(4))
    im = nilearn.image.resample_img(im, target_affine=target_affine)
  #  im = nib.load(im).get_data()
  #  nib.save(im, os.path.join('.','im.nii'))
  #  im = nib.load('im.nii')
    
   # print(im.shape)
    return im,im.get_data()

print(resample(im2)[1].shape)
print(resample(im2)[0].affine)


# In[3]:


im = 'patientIms/1/4DPWI.nii.gz'
im = nib.load(im).get_data()
print(im.shape)
print (np.min(im), np.max(im))


# In[ ]:




