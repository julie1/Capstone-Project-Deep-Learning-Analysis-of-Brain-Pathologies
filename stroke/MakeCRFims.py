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


def niiToNumpyCRFinput(path0, path1):
    im0 = nib.load(path0).get_data()
    im1 = nib.load(path1).get_data()
    assert im0.size == im1.size
    condlist = [np.add(im0, im1) < 0.90, np.add(im0, im1) > 0.90]
    choicelist = [1.0, im0]
    im0 = np.select(condlist, choicelist)
    
    affine = nib.load(path0).affine
    ims = np.stack([im0, im1], axis=0)
    
    return ims, im0.size, ims.shape[0], affine
path = 'examples/bestNoValid/outputTests/'
path0 = path + 'patient_47_ProbMapClass0.nii.gz'
path1 = path + 'patient_47_ProbMapClass1.nii.gz'

ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
print (npoints, nlabels, affine)
#ims = np.argmax(ims, axis=0).reshape(ims.shape[1:])



d = dcrf.DenseCRF(npoints, nlabels)  # npoints, nlabels
U =  ims
print(U.shape)        # -> (2, 128, 128, 25)
print(U.dtype)        # -> dtype('float32')
U = unary_from_softmax(ims)
d.setUnaryEnergy(U)


# In[3]:


path = 'examples/bestNoValid/outputTests/'
CRFpath = 'examples/outputCRF/'
pathG = 'examples/bestNoValid/inputTests/'
pathROI = 'examples/bestNoValid/inputTests/'
def run_CRF(patient, sdimsB, sdimsG, schan):
    
    
    path0 = path + 'patient_' + str(patient) +'_ProbMapClass0.nii.gz'
    path1 = path + 'patient_' + str(patient) +'_ProbMapClass1.nii.gz'
    ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
    shape1 = ims.shape[1:]
    
    name = 'patient_' + str(patient) + '_CRF'
    d = dcrf.DenseCRF(npoints, nlabels)  # npoints, nlabels
    U =  ims
    shape = U.shape[1:]
   # print (np.sum(U))
    U = unary_from_softmax(ims)
    d.setUnaryEnergy(U)
    G = create_pairwise_gaussian(sdimsG, shape)
    d.addPairwiseEnergy(w2*G, compat=compat1)
    B = create_pairwise_bilateral(sdimsB, schan, ims, chdim=0)
    d.addPairwiseEnergy(w1*B, compat=compat2)
 #   Q = d.inference(1)
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
       # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
    patientCRF = np.argmax(Q, axis=0).reshape(shape1).astype(np.float32)
    
    #print (patientCRF.shape, patientCRF.dtype, np.sum(patientCRF))
  #  if patient == 48:
    im = nib.Nifti1Image(patientCRF, affine=affine)
    nib.save(im, os.path.join(CRFpath, name + '.nii.gz'))
       # im1 = nib.load(CRFpath + name + '.nii.gz')
      #  print(im1.get_data().shape)
     #   print(im1.get_data().dtype)
     #   print(im1.header)
    return patientCRF

def groundniiToNumpyCRFinput(patient, pathG):
    
    
    path = pathG + 'patient_' + str(patient) + '/' + 'OT.nii.gz'
    im = nib.load(path).get_data()
        
    return im

def ROInii(patient, pathROI):
    path = pathROI + 'patient_' + str(patient) + '/' + 'brainmask.nii.gz'
    im = nib.load(path).get_data()
        
    return im

    
    


# In[4]:


def CRF_dice(patientCRF,patientGround,roi):
    
    save_dir = "checkpoint_bn"
    logs_path ="logs_bn"
    batch_size = 1
   # tl.files.exists_or_mkdir(save_dir)
  #  tl.files.exists_or_mkdir(logs_path)
     ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there is 1 label in targets:
    # Label 0: background
    # Label 1: stroke lesion
    
    
    X_valid = np.expand_dims(patientCRF, axis=0)
   # print ("X_valid:", X_valid.shape)
    y_valid = np.expand_dims(patientGround, axis=0)
  #  print ("y_valid:", y_valid.shape)
    ROI = np.expand_dims(roi, axis=0)
        
  #  X_valid = (X_valid > 0).astype(int)
  #  y_valid = (y_valid > 0).astype(int)
    nw, nh, nz = X_valid.shape[1:]
    
        
   
    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFINE MODEL =======================###
            
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='CRF_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, nz], name='target_segment')
            
            
            
            ###======================== DEFINE LOSS =========================###
            

            ## test losses
            test_out_seg = t_image
            test_dice_loss = 1 - tl.cost.dice_coe(t_image, t_seg, axis= (1,2,3))#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(t_image, t_seg, axis=(1,2,3))
           # test_dice_hard = tl.cost.dice_hard_coe(t_image, t_seg, threshold = 0.01, axis=(1,2,3))
            test_dice_hard = tl.cost.dice_hard_coe(t_image, t_seg, axis=(1,2,3))
            
            # create a summary for our cost and accuracy
            tf.summary.scalar("test_dice", test_dice_loss)
            tf.summary.scalar("test_hard-dice", test_dice_hard)
            tf.summary.scalar("test_iou", test_iou_loss)
            

            # merge all summaries into a single "operation" which we can execute in a session 
            summary_op = tf.summary.merge_all()

        
        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        
        
       # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())    
            
            ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        
        for batch in tl.iterate.minibatches(inputs=X_valid*roi, targets=y_valid*roi,
                                        batch_size=batch_size, shuffle=False):
            b_images, b_labels = batch
           # dp_dict = tl.utils.dict_to_one( net_test.all_drop )
            feed_dict = {t_image: b_images, t_seg: b_labels}
          #  feed_dict.update(dp_dict)
            _dice, _iou, _diceh = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard],
                    feed_dict=feed_dict)
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

       # print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f " %
               # (total_dice, total_dice_hard, total_iou))
        return (total_dice, total_dice_hard, total_iou)
    
    
    
    


# In[5]:


def patientNameToNum(cfg):
    patients = []
    f = open(cfg)    
    for line in f:        
        line = line.replace("patient_", "")
        line = line.replace(".nii.gz", "")        
        line = int(line)        
        patients.append(line)
    return patients


parameters = {'theta_x2': 4, 'compat1': 0.01, 'theta_chan0': 21, 'compat2': 10.0, 'theta_z1': 25,
                       'theta_z2': 44, 'theta_y2': 43, 'theta_y1': 32, 'theta_x1': 61, 'w2': 17, 'w1': 20,
                       'theta_chan1': 4}
#avg-dice: 0.6690174204962595
#avg-hard-dice: 0.3309825793171354
#avg-iou 0.22593376753551608



sdimsB = tuple([parameters[i] for i in ('theta_x1', 'theta_y1', 'theta_z1')])
sdimsG = tuple([parameters[i] for i in ('theta_x2', 'theta_y2', 'theta_z2')])
schan = tuple([parameters[i] for i in ('theta_chan0', 'theta_chan1')])
w1, w2 = tuple([parameters[i] for i in ('w1', 'w2')])
compat1, compat2 = parameters['compat1'], parameters['compat2']
print (sdimsB)


patient_path = "examples/configFiles/deepMedic/test/"
doc = "testPredictionNames.cfg"
patients = []
for i in range(5):
    
    patients = patients + patientNameToNum(patient_path+str(i)+'/'+doc)



print(patients)









total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
hd = 0
nh_batch = 0
for patient in patients:
    
    patientCRF = run_CRF(patient, sdimsB, sdimsG, schan).astype(np.uint64)
    patientGround = groundniiToNumpyCRFinput(patient, pathG).astype(np.uint64)
    roi = ROInii(patient, pathROI).astype(np.uint64)
    m1, m2, m3 = CRF_dice(patientCRF,patientGround,roi)
    print('patient:',  patient)
    print ('dice-loss:', m1)
    print ('hard-dice:', m2)
    print ('iou:', m3)
    total_dice += m1 
    total_dice_hard += m2 
    total_iou += m3
    surface_distances = sd.compute_surface_distances(patientCRF*roi, patientGround*roi, [1,1,1])
    s3 = sd.compute_robust_hausdorff(surface_distances, 95)
    
    if s3 != np.Inf: hd += s3    
    
    print('patient:', patient,  'hausdorff distance:', s3)

    if s3 != np.Inf: nh_batch += 1
    
    n_batch += 1
           
print ('avg-dice:', total_dice/n_batch)           
print ('avg-hard-dice:', total_dice_hard/n_batch)
print ('avg-iou', total_iou/n_batch)


print('average hausdorff distance', hd/nh_batch)
  


# In[2]:


#make CRF test images

def niiToNumpyCRFinput(path0, path1):
    im0 = nib.load(path0).get_data()
    im1 = nib.load(path1).get_data()
    assert im0.size == im1.size
    condlist = [np.add(im0, im1) < 0.90, np.add(im0, im1) > 0.90]
    choicelist = [1.0, im0]
    im0 = np.select(condlist, choicelist)
    
    affine = nib.load(path0).affine
    ims = np.stack([im0, im1], axis=0)
    
    return ims, im0.size, ims.shape[0], affine


# In[3]:


#make CRF test images

path = 'examples/outputTest/predictions/testSessionDeepMedicStrokes/predictions/'
CRFpath = 'examples/outputTest/outputCRF/'
#pathG = 'examples/bestNoValid/inputTests/'
pathROI = 'examples/dataForStrokes/test/'
def run_CRF(patient, sdimsB, sdimsG, schan):
    
    
    path0 = path + 'patient_' + str(patient) +'_ProbMapClass0.nii.gz'
    path1 = path + 'patient_' + str(patient) +'_ProbMapClass1.nii.gz'
    ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
    shape1 = ims.shape[1:]
    
    name = 'patient_' + str(patient) + '_CRF'
    d = dcrf.DenseCRF(npoints, nlabels)  # npoints, nlabels
    U =  ims
    shape = U.shape[1:]
   # print (np.sum(U))
    U = unary_from_softmax(ims)
    d.setUnaryEnergy(U)
    G = create_pairwise_gaussian(sdimsG, shape)
    d.addPairwiseEnergy(w2*G, compat=compat1)
    B = create_pairwise_bilateral(sdimsB, schan, ims, chdim=0)
    d.addPairwiseEnergy(w1*B, compat=compat2)
 #   Q = d.inference(1)
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
       # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
    patientCRF = np.argmax(Q, axis=0).reshape(shape1).astype(np.float32)
    
    #print (patientCRF.shape, patientCRF.dtype, np.sum(patientCRF))
  #  if patient == 48:
    im = nib.Nifti1Image(patientCRF, affine=affine)
    nib.save(im, os.path.join(CRFpath, name + '.nii.gz'))
       # im1 = nib.load(CRFpath + name + '.nii.gz')
      #  print(im1.get_data().shape)
     #   print(im1.get_data().dtype)
     #   print(im1.header)
    return patientCRF



def ROInii(patient, pathROI):
    path = pathROI + 'patient_' + str(patient) + '/' + 'brainmask.nii.gz'
    im = nib.load(path).get_data()
        
    return im

    
    


# In[4]:


#make CRF test images


def patientNameToNum(cfg):
    patients = []
    f = open(cfg)    
    for line in f:        
        line = line.replace("patient_", "")
        line = line.replace(".nii.gz", "")        
        line = int(line)        
        patients.append(line)
    return patients


parameters = {'theta_x2': 4, 'compat1': 0.01, 'theta_chan0': 21, 'compat2': 10.0, 'theta_z1': 25,
                       'theta_z2': 44, 'theta_y2': 43, 'theta_y1': 32, 'theta_x1': 61, 'w2': 17, 'w1': 20,
                       'theta_chan1': 4}
#avg-dice: 0.6690174204962595
#avg-hard-dice: 0.3309825793171354
#avg-iou 0.22593376753551608



sdimsB = tuple([parameters[i] for i in ('theta_x1', 'theta_y1', 'theta_z1')])
sdimsG = tuple([parameters[i] for i in ('theta_x2', 'theta_y2', 'theta_z2')])
schan = tuple([parameters[i] for i in ('theta_chan0', 'theta_chan1')])
w1, w2 = tuple([parameters[i] for i in ('w1', 'w2')])
compat1, compat2 = parameters['compat1'], parameters['compat2']
print (sdimsB)


patient_path = "examples/configFiles/deepMedic/testIsles2017"
doc = "testPredictionNames.cfg"
patients = []

    
patients = patients + patientNameToNum(patient_path+'/'+doc)



print(patients)










for patient in patients:
    
    patientCRF = run_CRF(patient, sdimsB, sdimsG, schan).astype(np.uint64)
    
    


# In[2]:


# rename test CRF images according to ISLES convention SMIR.my_result_01.129319.nii

CRFpath = 'examples/outputTest/outputCRF/'
Islespath = "U-net_code_stroke/ISLES2017_Testing/"
IslesCRFpath = 'examples/outputTest/outputCRFIsles2017/'
patients = [15, 4, 36, 3, 8, 37, 18, 19, 16, 34, 20, 14, 1, 6, 39, 32, 10, 2, 21, 30, 27, 5, 28, 35, 7,
            11, 17, 31, 38, 29, 33, 40]


for p in patients:
    CRFpatientPath = CRFpath + 'patient_' +str(p) +'_CRF.nii.gz'
    patient_path = Islespath + 'test_' + str(p) + '/'
    pMri = glob(os.path.join(patient_path, '*.Brain.XX.O.MR_' + 'MTT' +'.*'))[0]
    print(pMri)
    ID = pMri[pMri.find('MTT.')+4:]
    print (p, ID)
    if p < 10:
        islesName = 'SMIR.my_result_0' + str(p) + '.' + ID + '.nii'
    else: islesName = 'SMIR.my_result_' + str(p) + '.' + ID + '.nii'
    print (islesName)
    img = nib.load(CRFpatientPath)
    img.set_data_dtype(np.uint8)
   # img_data = img.get_data().astype(np.uint8)
    nib.save(img, IslesCRFpath + islesName)
  #  src = os.path.join(CRFpath, 'patient_' +str(p) +'_CRF.nii.gz')
  #  dst = os.path.join(IslesCRFpath , islesName)
  #  print(nib.load(src).header)
           
  #  shutil.copyfile(src, dst)    
    
  #  print (nib.load(dst).header)


# In[3]:


impath = 'examples/outputTest/outputCRFIsles2017/' + 'SMIR.my_result_15.129347.nii'

im = nib.load(impath)
print(im.header)
print(im.get_data().dtype)


# In[2]:


#make CRF train images

def niiToNumpyCRFinput(path0, path1):
    im0 = nib.load(path0).get_data()
    im1 = nib.load(path1).get_data()
    assert im0.size == im1.size
    condlist = [np.add(im0, im1) < 0.90, np.add(im0, im1) > 0.90]
    choicelist = [1.0, im0]
    im0 = np.select(condlist, choicelist)
    
    affine = nib.load(path0).affine
    ims = np.stack([im0, im1], axis=0)
    
    return ims, im0.size, ims.shape[0], affine
path = 'examples/bestNoValid/outputTests/'
path0 = path + 'patient_47_ProbMapClass0.nii.gz'
path1 = path + 'patient_47_ProbMapClass1.nii.gz'

ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
print (npoints, nlabels, affine)
#ims = np.argmax(ims, axis=0).reshape(ims.shape[1:])



d = dcrf.DenseCRF(npoints, nlabels)  # npoints, nlabels
U =  ims
print(U.shape)        # -> (2, 128, 128, 25)
print(U.dtype)        # -> dtype('float32')
U = unary_from_softmax(ims)
d.setUnaryEnergy(U)


# In[3]:


#make CRF train images

path = 'examples/output/predictions/testSessionDeepMedicStrokes/predictions/'
CRFpath = 'examples/output/outputCRFtrain/'
pathG = 'examples/dataForStrokes/test/'
pathROI = 'examples/dataForStrokes/test/'
def run_CRF(patient, sdimsB, sdimsG, schan):
    
    
    path0 = path + 'patient_' + str(patient) +'_ProbMapClass0.nii.gz'
    path1 = path + 'patient_' + str(patient) +'_ProbMapClass1.nii.gz'
    ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
    shape1 = ims.shape[1:]
    
    name = 'patient_' + str(patient) + '_CRF'
    d = dcrf.DenseCRF(npoints, nlabels)  # npoints, nlabels
    U =  ims
    shape = U.shape[1:]
   # print (np.sum(U))
    U = unary_from_softmax(ims)
    d.setUnaryEnergy(U)
    G = create_pairwise_gaussian(sdimsG, shape)
    d.addPairwiseEnergy(w2*G, compat=compat1)
    B = create_pairwise_bilateral(sdimsB, schan, ims, chdim=0)
    d.addPairwiseEnergy(w1*B, compat=compat2)
 #   Q = d.inference(1)
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
       # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
    patientCRF = np.argmax(Q, axis=0).reshape(shape1).astype(np.float32)
    
    #print (patientCRF.shape, patientCRF.dtype, np.sum(patientCRF))
  #  if patient == 48:
    im = nib.Nifti1Image(patientCRF, affine=affine)
    nib.save(im, os.path.join(CRFpath, name + '.nii.gz'))
       # im1 = nib.load(CRFpath + name + '.nii.gz')
      #  print(im1.get_data().shape)
     #   print(im1.get_data().dtype)
     #   print(im1.header)
    return patientCRF

def groundniiToNumpyCRFinput(patient, pathG):
    
    
    path = pathG + 'patient_' + str(patient) + '/' + 'OT.nii.gz'
    im = nib.load(path).get_data()
        
    return im

def ROInii(patient, pathROI):
    path = pathROI + 'patient_' + str(patient) + '/' + 'brainmask.nii.gz'
    im = nib.load(path).get_data()
        
    return im

    
    


# In[4]:


#make CRF train images

def CRF_dice(patientCRF,patientGround,roi):
    
    save_dir = "checkpoint_bn"
    logs_path ="logs_bn"
    batch_size = 1
   # tl.files.exists_or_mkdir(save_dir)
  #  tl.files.exists_or_mkdir(logs_path)
     ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there is 1 label in targets:
    # Label 0: background
    # Label 1: stroke lesion
    
    
    X_valid = np.expand_dims(patientCRF, axis=0)
   # print ("X_valid:", X_valid.shape)
    y_valid = np.expand_dims(patientGround, axis=0)
  #  print ("y_valid:", y_valid.shape)
    ROI = np.expand_dims(roi, axis=0)
        
  #  X_valid = (X_valid > 0).astype(int)
  #  y_valid = (y_valid > 0).astype(int)
    nw, nh, nz = X_valid.shape[1:]
    
        
   
    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFINE MODEL =======================###
            
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='CRF_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, nz], name='target_segment')
            
            
            
            ###======================== DEFINE LOSS =========================###
            

            ## test losses
            test_out_seg = t_image
            test_dice_loss = 1 - tl.cost.dice_coe(t_image, t_seg, axis= (1,2,3))#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(t_image, t_seg, axis=(1,2,3))
           # test_dice_hard = tl.cost.dice_hard_coe(t_image, t_seg, threshold = 0.01, axis=(1,2,3))
            test_dice_hard = tl.cost.dice_hard_coe(t_image, t_seg, axis=(1,2,3))
            
            # create a summary for our cost and accuracy
            tf.summary.scalar("test_dice", test_dice_loss)
            tf.summary.scalar("test_hard-dice", test_dice_hard)
            tf.summary.scalar("test_iou", test_iou_loss)
            

            # merge all summaries into a single "operation" which we can execute in a session 
            summary_op = tf.summary.merge_all()

        
        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        
        
       # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())    
            
            ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        
        for batch in tl.iterate.minibatches(inputs=X_valid*roi, targets=y_valid*roi,
                                        batch_size=batch_size, shuffle=False):
            b_images, b_labels = batch
           # dp_dict = tl.utils.dict_to_one( net_test.all_drop )
            feed_dict = {t_image: b_images, t_seg: b_labels}
          #  feed_dict.update(dp_dict)
            _dice, _iou, _diceh = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard],
                    feed_dict=feed_dict)
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

       # print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f " %
               # (total_dice, total_dice_hard, total_iou))
        return (total_dice, total_dice_hard, total_iou)
    
    
    
    


# In[5]:


#make CRF train images

def patientNameToNum(cfg):
    patients = []
    f = open(cfg)    
    for line in f:        
        line = line.replace("patient_", "")
        line = line.replace(".nii.gz", "")        
        line = int(line)        
        patients.append(line)
    return patients


parameters = {'theta_x2': 4, 'compat1': 0.01, 'theta_chan0': 21, 'compat2': 10.0, 'theta_z1': 25,
                       'theta_z2': 44, 'theta_y2': 43, 'theta_y1': 32, 'theta_x1': 61, 'w2': 17, 'w1': 20,
                       'theta_chan1': 4}
#avg-dice: 0.6690174204962595
#avg-hard-dice: 0.3309825793171354
#avg-iou 0.22593376753551608



sdimsB = tuple([parameters[i] for i in ('theta_x1', 'theta_y1', 'theta_z1')])
sdimsG = tuple([parameters[i] for i in ('theta_x2', 'theta_y2', 'theta_z2')])
schan = tuple([parameters[i] for i in ('theta_chan0', 'theta_chan1')])
w1, w2 = tuple([parameters[i] for i in ('w1', 'w2')])
compat1, compat2 = parameters['compat1'], parameters['compat2']
print (sdimsB)


patient_path = "examples/configFiles/deepMedic/trainIsles2017/"
doc = "testPredictionNames.cfg"
patients = []

    
patients = patients + patientNameToNum(patient_path+'/'+doc)



print(patients)









total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
hd = 0
nh_batch = 0
for patient in patients:
    
    patientCRF = run_CRF(patient, sdimsB, sdimsG, schan).astype(np.uint64)
    patientGround = groundniiToNumpyCRFinput(patient, pathG).astype(np.uint64)
    roi = ROInii(patient, pathROI).astype(np.uint64)
    m1, m2, m3 = CRF_dice(patientCRF,patientGround,roi)
    print('patient:',  patient)
    print ('dice-loss:', m1)
    print ('hard-dice:', m2)
    print ('iou:', m3)
    total_dice += m1 
    total_dice_hard += m2 
    total_iou += m3
    surface_distances = sd.compute_surface_distances(patientCRF*roi, patientGround*roi, [1,1,1])
    s3 = sd.compute_robust_hausdorff(surface_distances, 95)
    
    if s3 != np.Inf: hd += s3    
    
    print('patient:', patient,  'hausdorff distance:', s3)

    if s3 != np.Inf: nh_batch += 1
    
    n_batch += 1
           
print ('avg-dice:', total_dice/n_batch)           
print ('avg-hard-dice:', total_dice_hard/n_batch)
print ('avg-iou', total_iou/n_batch)


print('average hausdorff distance', hd/nh_batch)
  


# In[6]:


# rename train CRF images according to ISLES convention SMIR.my_result_01.129319.nii

CRFpath = 'examples/output/outputCRFtrain/'
Islespath = "U-net_code_stroke/ISLES2017_Training/"
IslesCRFpath = 'examples/output/outputCRFtrainIsles2017/'
patients = [27, 10, 41, 42, 6, 47, 35, 44, 32, 9, 16, 19, 43, 7, 24, 14, 45, 23, 30, 20, 46, 21, 22, 38,
            12, 8, 15, 48, 26, 33, 1, 31, 11, 5, 18, 40, 36, 39, 28, 37, 13, 2, 4]

for p in patients:
    CRFpatientPath = CRFpath + 'patient_' +str(p) +'_CRF.nii.gz'
    patient_path = Islespath + 'training_' + str(p) + '/'
    pMri = glob(os.path.join(patient_path, '*.Brain.XX.O.MR_' + 'MTT' +'.*'))[0]
    print(pMri)
    ID = pMri[pMri.find('MTT.')+4:]
    print (p, ID)
    if p < 10:
        islesName = 'SMIR.my_result_0' + str(p) + '.' + ID + '.nii'
    else: islesName = 'SMIR.my_result_' + str(p) + '.' + ID + '.nii'
    print (islesName)
    img = nib.load(CRFpatientPath)
    img.set_data_dtype(np.uint8)
   # img_data = img.get_data().astype(np.uint8)
    nib.save(img, IslesCRFpath + islesName)
  #  src = os.path.join(CRFpath, 'patient_' +str(p) +'_CRF.nii.gz')
  #  dst = os.path.join(IslesCRFpath , islesName)
  #  print(nib.load(src).header)
           
  #  shutil.copyfile(src, dst)    
    
  #  print (nib.load(dst).header)


# In[ ]:




