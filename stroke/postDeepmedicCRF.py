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
from scipy.stats import uniform

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
  #  im = nib.Nifti1Image(patientCRF, affine=affine)
  #  nib.save(im, os.path.join(CRFpath, name + '.nii.gz'))
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

    
    


# In[3]:


parameters = {'theta_x2': 0.6504359397524828, 'compat1': 2.0480420657858645, 'theta_chan0': 0.606257308096001,
              'compat2': 3.142265623598287, 'theta_z1': 0.034273674210661964, 'theta_z2': 1.9238795729035252,
              'theta_y2': 6.47626355024311, 'theta_y1': 2.632054064435574, 'theta_x1': 6.404915980144311,
              'w2': 4.556037558505677, 'w1': 3.3812039213138174, 'theta_chan1': 8.503513237032548}




sdimsB = tuple([parameters[i] for i in ('theta_x1', 'theta_y1', 'theta_z1')])
sdimsG = tuple([parameters[i] for i in ('theta_x2', 'theta_y2', 'theta_z2')])
schan = sum(tuple([parameters[i] for i in ('theta_chan0', 'theta_chan1')]))/2
compat1, compat2 = tuple([parameters[i] for i in ('compat1', 'compat2')])
w1, w2 = tuple([parameters[i] for i in ('w1', 'w2')])
print (sdimsB)


# In[9]:


def patientNameToNum(cfg):
    patients = []
    f = open(cfg)    
    for line in f:        
        line = line.replace("patient_", "")
        line = line.replace(".nii.gz", "")        
        line = int(line)        
        patients.append(line)
    return patients


patient_path = 'examples/configFiles/deepMedic/test/'
doc = 'testPredictionNames.cfg'
patients = []
for i in range(5):
    
    patients = patients + patientNameToNum(patient_path+str(i)+'/'+doc)
print(patients)
#for patient in patients:
   # run_CRF(patient, sdimsB, sdimsG, schan)


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


patient_path = "examples/configFiles/deepMedic/test/"
doc = "testPredictionNames.cfg"
patients = []
for i in range(5):
    
    patients = patients + patientNameToNum(patient_path+str(i)+'/'+doc)


#patient_path = "examples/configFiles/deepMedic/train/trainPredictionNames.cfg"
#patients = patientNameToNum(patient_path)
print(patients)


test_dice = []
hard_dice = []
iou = []
#comp = np.random.uniform(low=0,high=10, size=(50,2))
w = np.random.randint(1, 21, size=(100,2))
theta = np.random.randint(1, 81, size=(100,8))
exp1 = np.random.randint(-5, 1, size=100).astype(float)
exp2 = np.random.randint(-2, 3, size=100).astype(float)
#compat1, compat2 = 1, 1

parameters = []

names = ["compat1","compat2","w1","w2","theta_x1", "theta_y1","theta_z1", "theta_x2", "theta_y2", "theta_z2",
         "theta_chan0", "theta_chan1"]
#select parameters from a random distribution
for i in range(100):    
    
   # compat1, compat2 = comp[i][0], comp[i][1]
    w1, w2 = w[i][0], w[i][1]    
    theta_x1, theta_y1, theta_z1, theta_x2, theta_y2, theta_z2, theta_chan0, theta_chan1 = tuple([theta[i, j] for j in range(8)])
    sdimsB = (theta_x1, theta_y1, theta_z1)
    sdimsG = (theta_x2, theta_y2, theta_z2)
    schan = (theta_chan0, theta_chan1)
    e1 = exp1[i]
    e2 = exp2[i]
        
    compat1 = 10**e1
    compat2 = 10**e2
  #  compat = (compat1, compat2) 
      
    parameters.append([compat1, compat2,w1, w2, theta_x1, theta_y1, theta_z1, theta_x2, theta_y2, theta_z2, theta_chan0, theta_chan1])
    print ("epoch:", i, "parameters:", parameters[i])
    total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
    for patient in patients:
      #  if i == 0: print(patient)
        patientCRF = run_CRF(patient, sdimsB, sdimsG, schan)
        patientGround = groundniiToNumpyCRFinput(patient, pathG)
        roi = ROInii(patient, pathROI)
        m1, m2, m3 = CRF_dice(patientCRF,patientGround,roi)
        total_dice += m1 
        total_dice_hard += m2 
        total_iou += m3 
        n_batch += 1
        print(" **"+" "*5+"patient: %d  test 1-dice: %f hard-dice: %f iou: %f " %
                (patient, m1, m2, m3))
    test_dice.append(total_dice/n_batch)
    hard_dice.append(total_dice_hard/n_batch)
    iou.append(total_iou/n_batch)
    print(" **"+" "*5+"epoch: %d  average test 1-dice: %f average hard-dice: %f average iou: %f " %
                (i, test_dice[i], hard_dice[i], iou[i]))
test_dice = np.array(test_dice)
hard_dice = np.array(hard_dice)
iou = np.array(iou)
ind0 = np.argmin(test_dice)
ind1 = np.argmax(hard_dice)
ind2 = np.argmax(iou)
inds0 = np.argsort(test_dice)[:10]
inds1 = np.argsort(hard_dice)[::-1][:10]
inds2 = np.argsort(iou)[::-1][:10]
 
parameters0 = dict(zip(names,parameters[ind0]))                        
parameters1 = dict(zip(names,parameters[ind1]))                        
parameters2 = dict(zip(names,parameters[ind2]))
print (ind0, "test_dice_best:", test_dice[ind0])
print (parameters0)
print (ind1, "hard_dice_best:", hard_dice[ind1])
print (parameters1)
print (ind2, "iou_best:", iou[ind2])
print (parameters2)
print("test_dice:", inds0, [(test_dice[j], dict(zip(names,parameters[j]))) for j in inds0 ])
print("hard_dice:", inds1, [(hard_dice[j], dict(zip(names,parameters[j])))for j in inds1 ])
print("iou:", inds2, [(iou[j], dict(zip(names,parameters[j])))for j in inds2 ])
                        


# In[ ]:


hard_dice: [34 66 59 12 36 27 18 53 56 77]
names = ["compat1","compat2","w1","w2","theta_x1", "theta_y1","theta_z1", "theta_x2", "theta_y2", "theta_z2",
         "theta_chan0", "theta_chan1"]
epoch: 34 parameters: [0.1, 10.0, 20, 17, 61, 32, 25, 5, 42, 44, 20, 4]
epoch: 66 parameters: [1.0, 1.0, 8, 11, 3, 28, 4, 75, 35, 13, 26, 75]
epoch: 59 parameters: [1.0, 1.0, 16, 17, 73, 47, 70, 41, 60, 53, 5, 47]
epoch: 12 parameters: [0.01, 100.0, 15, 9, 24, 9, 11, 79, 38, 2, 39, 3]
epoch: 36 parameters: [0.001, 1.0, 8, 16, 41, 15, 4, 37, 15, 31, 43, 76]


# In[ ]:


comp = np.random.uniform(low=0,high=10, size=(10,2))

w = np.random.uniform(low=0, high=5, size=(10,2))

theta = np.random.uniform(low=0, high=10, size=(10,8)) 

for i in range(10):
    parameters = []
    print(i)
    compat1, compat2 = comp[i][0], comp[i][1]
    w1, w2 = w[i][0], w[i][1]
    
    theta_x1, theta_y1, theta_z1, theta_x2, theta_y2, theta_z2, theta_chan0, theta_chan1 = tuple([theta[i, j] for j in range(8)])
    sdimsB = (theta_x1, theta_y1, theta_z1)
    sdimsG = (theta_x2, theta_y2, theta_z2)
    schan = (theta_chan0, theta_chan1)
    compat = (compat1, compat2)    
    parameters.extend([compat1, compat2, w1, w2, theta_x1, theta_y1, theta_z1, theta_x2, theta_y2, theta_z2, theta_chan0, theta_chan1])
    print (parameters)


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
path = 'examples/outputPretrain/outputPretrainAllbest0/predictions/testSessionDeepMedicStrokes/predictions/'
path0 = path + 'patient_47_ProbMapClass0.nii.gz'
path1 = path + 'patient_47_ProbMapClass1.nii.gz'

ims, npoints, nlabels, affine = niiToNumpyCRFinput(path0, path1)
print (npoints, nlabels, affine)
#ims = np.argmax(ims, axis=0).reshape(ims.shape[1:])


# In[3]:


path = 'examples/output/predictions/testSessionDeepMedicStrokes_beforeCRF/predictions/'
CRFpath = 'examples/output/CRF/'
pathG = 'examples/dataForStrokes/train/0/'
pathROI = 'examples/dataForStrokes/train/0/'

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
            test_dice_loss = 1 - tl.cost.dice_coe(t_image, t_seg, axis=[1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(t_image, t_seg, axis=[1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(t_image, t_seg, axis=[1,2,3])
            
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
        
        for batch in tl.iterate.minibatches(inputs=X_valid*roi, targets=y_valid,
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
           #     (total_dice, total_dice_hard, total_iou))
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


parameters = {'theta_chan0': 2.8883564788836025, 'theta_x2': 55.27420282567868, 'theta_z1': 8.060314635415777, 
              'theta_z2': 10.78111199509303, 'theta_y2': 31.618253880299186, 'theta_y1': 41.914678693727645,
              'theta_x1': 57.96871876797912, 'w2': 9.892769884270024, 'w1': 9.688848117511178, 
              'theta_chan1': 17.745886967131298}



sdimsB = tuple([parameters[i] for i in ('theta_x1', 'theta_y1', 'theta_z1')])
sdimsG = tuple([parameters[i] for i in ('theta_x2', 'theta_y2', 'theta_z2')])
schan = tuple([parameters[i] for i in ('theta_chan0', 'theta_chan1')])
w1, w2 = tuple([parameters[i] for i in ('w1', 'w2')])
print (sdimsB)


patient_path = "examples/configFiles/deepMedic/train/trainPredictionNames.cfg"
patients = patientNameToNum(patient_path)
print(patients)



compat1, compat2 = 1, 1




total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
for patient in patients:
    
    patientCRF = run_CRF(patient, sdimsB, sdimsG, schan)
    patientGround = groundniiToNumpyCRFinput(patient, pathG)
    roi = ROInii(patient, pathROI)
    m1, m2, m3 = CRF_dice(patientCRF,patientGround,roi)
    print('patient:',  patient)
    print ('dice-loss:', m1)
    print ('hard-dice:', m2)
    print ('iou:', m3)
    total_dice += m1 
    total_dice_hard += m2 
    total_iou += m3 
    n_batch += 1
           
print ('avg-dice:', total_dice/n_batch)           
print ('avg-hard-dice:', total_dice_hard/n_batch)
print ('avg-iou', total_iou/n_batch)
           
  


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


def groundniiToNumpyCRFinput(patient, pathG):
    
    
    path = pathG + 'patient_' + str(patient) + '/' + 'OT.nii.gz'
    im = nib.load(path).get_data()
        
    return im

def ROInii(patient, pathROI):
    path = pathROI + 'patient_' + str(patient) + '/' + 'brainmask.nii.gz'
    im = nib.load(path).get_data()
        
    return im


patient_path = 'examples/configFiles/deepMedic/test/'
doc = 'testPredictionNames.cfg'
patients = []
for i in range(5):
    
    patients = patients + patientNameToNum(patient_path+str(i)+'/'+doc)


print(patients)
path = 'examples/bestNoValid/outputTests/'
CRFpath = 'examples/outputCRF/'
pathG = 'examples/bestNoValid/inputTests/'
pathROI = 'examples/bestNoValid/inputTests/'

test_dice = []
hard_dice = []
iou = []

total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
for patient in patients:
    
    patientSegm = nib.load(Segm_path+'patient_'+str(patient)+'_Segm.nii.gz').get_data()    
    patientGround = groundniiToNumpyCRFinput(patient, pathG)
    roi = ROInii(patient, pathROI)
    m1, m2, m3 = CRF_dice(patientSegm,patientGround,roi)
    total_dice += m1 
    total_dice_hard += m2 
    total_iou += m3 
    n_batch += 1
    test_dice.append((patient,m1))
    hard_dice.append((patient,m2))
    iou.append((patient,m3))
    print('patient:', patient,  'test dice:', m1, 'hard dice:', m2, 'iou:', m3)
print(n_batch)    
avg_dice = total_dice/n_batch
avg_hard_dice = total_dice_hard/n_batch
avg_iou = total_iou/n_batch
print ('average dice:', avg_dice)
print('average hard dice:', avg_hard_dice)
print('average iou:', avg_iou)
print(test_dice)
print(hard_dice)
print(iou)


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

patient_path = "examples/configFiles/deepMedic/test/"
doc = "testPredictionNames.cfg"
patients = []
for i in range(5):
    
    patients = patients + patientNameToNum(patient_path+str(i)+'/'+doc)


#patient_path = "examples/outputCRF/configFilesCRF/deepMedic/test/testPredictionNames.cfg"
#patients = patientNameToNum(patient_path)
print(patients)
CRFpath = "examples/outputCRF/CRF/"
pathG = 'examples/dataForStrokes/train/0/'
pathROI = 'examples/dataForStrokes/train/0/'

test_dice = []
hard_dice = []
iou = []

total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
for patient in patients:
    print(patient)
    patientCRF = nib.load(CRFpath+'patient_'+str(patient)+'_CRF.nii.gz').get_data()    
    patientGround = groundniiToNumpyCRFinput(patient, pathG)
    roi = ROInii(patient, pathROI)
    m1, m2, m3 = CRF_dice(patientCRF,patientGround,roi)
    total_dice += m1 
    total_dice_hard += m2 
    total_iou += m3 
    n_batch += 1
    test_dice.append((patient,m1))
    hard_dice.append((patient,m2))
    iou.append((patient,m3))
    print('patient:', patient,  'test dice:', m1, 'hard dice:', m2, 'iou:', m3)
    
avg_dice = total_dice/n_batch
avg_hard_dice = total_dice_hard/n_batch
avg_iou = total_iou/n_batch
print ('average dice:', avg_dice)
print('average hard dice:', avg_hard_dice)
print('average iou:', avg_iou)


# In[ ]:




