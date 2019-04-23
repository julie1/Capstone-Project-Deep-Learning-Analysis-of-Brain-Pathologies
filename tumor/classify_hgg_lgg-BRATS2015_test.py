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


from IPython.display import display, Image
from scipy import ndimage, misc

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from sklearn.model_selection import train_test_split

import h5py
import hdf5storage
import requests
import shutil
from urlparse import urlparse
from medpy.io import load
from medpy.io import save
import nibabel as nib
import SimpleITK as sitk
import json
import vtk


# In[2]:


#use stacked numpy arrays from previous training and testing runs to get one patient at
# a time to run through u-net for BRATS2015
maskpath = "/home/julie/U-net_code_Tumor/brainmasks/"

def get_HGG_train(i):

    if i < 66:
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_train'][i:i+1].tolist()
        #    print (path[0])
            
        with h5py.File('HGGstack_train1.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_train1'][155*i:155*(i+1),...]         
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_train'][155*i:155*(i+1),...]            
#             del hf

    if 66 <= i < 110:
        i1 = i - 66
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_valid'][i1:i1+1].tolist()
         #   print (path[0])
        with h5py.File('HGGstack_valid.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_valid'][155*i1:155*(i1+1),...]
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_valid'][155*i1:155*(i1+1),...]            
#             del hf

    if 110 <= i < 176:
        i2 = i - 44
        j = i - 110
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_train'][i2:i2+1].tolist()
         #   print (path[0])
        with h5py.File('HGGstack_train2.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_train2'][155*j:155*(j+1),...]         
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_train'][155*i2:155*(i2+1),...]            
#             del hf

    if 176 <= i < 220:
        i3 = i - 176
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_test'][i3:i3+1].tolist()
        #    print (path[0])
        with h5py.File('HGGstack_test.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_test'][155*i3:155*(i3+1),...]
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_test'][155*i3:155*(i3+1),...]            
#             del hf
            
#     l1 = path[0].find('Flair.')
#     l2 = path[0][l1+6:].find('Flair.')
#    # print(l1,l2)
#     path1 = path[0][l1+l2+6:]
#   #  print(path1)
#     return path, path1, X_train, y_train

    p1 = path[0].find('pat')
    p2 = path[0].find('/VSD')
    
  #  X_train = distort_imgs(X_train)
    brainmask = nib.load(os.path.join(maskpath+'HGG/'+path[0][p1:p2]+'/', 'brainmask.nii.gz')).get_data()
    brainmask = np.moveaxis(brainmask,-1,0)
    X_train = np.stack([X_train[:,:,:,i]*brainmask for i in range(4)], axis = -1)
    
    return ndimage.zoom(X_train,(0.41, 0.2, 0.2, 1) )
    

def get_LGG_train(i):
    if i < 32:
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsLGG_"+ 'Flair' + '_train'][i:i+1].tolist()
        #    print (path[0])
        with h5py.File('LGGstack_train.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_train'][155*i:155*(i+1),...]         
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_train'][155*i:155*(i+1),...]            
#             del hf

    if 32 <= i < 43:
        i1 = i - 32
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsLGG_"+ 'Flair' + '_valid'][i1:i1+1].tolist()
         #   print (path[0])
        with h5py.File('LGGstack_valid.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_valid'][155*i1:155*(i1+1),...]
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_valid'][155*i1:155*(i1+1),...]            
#             del hf
            
        
    if 43 <= i < 54:
        i3 = i - 43
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsLGG_"+ 'Flair' + '_test'][i3:i3+1].tolist()
          #  print (path[0])
        with h5py.File('LGGstack_test.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_test'][155*i3:155*(i3+1),...]
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_test'][155*i3:155*(i3+1),...]            
#             del hf
#     l1 = path[0].find('Flair.')
#     l2 = path[0][l1+6:].find('Flair.')
#     path1 = path[0][l1+l2+6:]
#     print(path1)
#     return path, path1, X_train, y_train

    
   # X_train = distort_imgs(X_train) 
    p1 = path[0].find('pat')
    p2 = path[0].find('/VSD')
   # print(path[0][p1:p2]) 
    brainmask = nib.load(os.path.join(maskpath+'LGG/'+path[0][p1:p2]+'/', 'brainmask.nii.gz')).get_data()
    brainmask = np.moveaxis(brainmask,-1,0)
    X_train = np.stack([X_train[:,:,:,i]*brainmask for i in range(4)], axis = -1)
   # print(X_train.shape)
    
  #  return path, path1, ndimage.zoom(X_train, 0.1)
    
   # if p == 1: X_train = np.stack([distort_imgs(X_train[n,:,:,:]) for n in range(X_train.shape[0])], axis=0)
  #  X_train = distort_imgs(X_train)   
   # print(X_train.shape)    
    
    return ndimage.zoom(X_train, (0.41, 0.2, 0.2, 1))
    

#path, path1, mri = get_HGG_train(0)
mri = get_LGG_train(0)
#print (path, path1, mri.shape)
print (mri.shape)
d, h, w, m = mri.shape


# In[3]:


#use stacked numpy arrays from previous training and testing runs to get one patient at
# a time to run through u-net for BRATS2015

maskpath_test = "/home/julie/U-net_code_Tumor/brainmasks_test/"
data_types = ['Flair', 'T1', 'T1c', 'T2']
def get_HGG_LGG_test(i):

    if i < 55:
        with h5py.File('PathsTest.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsTest"][0][i:i+1]
          #  print (path)
            
        with h5py.File('Teststack1.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_test = hf['Teststack1'][155*i:155*(i+1),...]         
            del hf
        

    if 55 <= i < 110:
        i1 = i - 55
        with h5py.File('PathsTest.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsTest"][0][i:i+1]
         #   print (path)
        with h5py.File('Teststack2.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_test = hf['Teststack2'][155*i1:155*(i1+1),...]
            del hf
       
    p1 = path[0].find('pat')
    p2 = path[0].find('/VSD')
  #  print(path[0][p1:p2]) 
    brainmask = nib.load(os.path.join(maskpath_test+'HGG_LGG/'+path[0][p1:p2]+'/', 'brainmask.nii.gz')).get_data()
    brainmask = np.moveaxis(brainmask,-1,0)
    X_test = np.stack([X_test[:,:,:,i]*brainmask for i in range(4)], axis = -1)
  #  print(X_test.shape)        
    
    
    return path[0][p1:p2], ndimage.zoom(X_test, (0.41, 0.2, 0.2, 1))


mri = get_HGG_LGG_test(0)
#print (path, path1, mri.shape)
print (mri[0], mri[1].shape)
d, h, w, m = mri[1].shape


# In[4]:


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
#print(labels[0], fetch_mri(0).shape)    
showDataImages(get_HGG_train(0)[:,:,:,0], 10, 5)
showDataImages(get_HGG_LGG_test(0)[1][:,:,:,2], 10, 5)


# In[4]:


from __future__ import print_function

from time import time
import logging
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.feature_selection import RFE
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274
numHGG = 220
mris = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)
labels = np.ndarray(shape=(num,), dtype = np.int16)
indices = np.random.permutation([i for i in range(labels.shape[0])])

for j in range(len(indices)):
   # print(j, indices[j])
    if indices[j] < numHGG: 
        mris[j,:,:,:] = get_HGG_train(indices[j])
        labels[j] = 0
    elif indices[j] >= numHGG: 
        mris[j,:,:,:] = get_LGG_train(indices[j]-numHGG)        
        labels[j] = 1    
    
        
# introspect the images arrays to find the shapes (for plotting)
n_samples, d, h, w, m = mris.shape
mris_flat = np.reshape(mris, (num, h*w*d*m))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

n_features = mris_flat.shape[1]

X_train, y_train = mris_flat, labels

# the label to predict is the id of the person
label_names = ['HG', 'LG']
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################
# Compute a PCA (eigenmris) on the mri dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 25

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
PCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = PCA.components_.reshape((n_components, h*16, w*16))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = PCA.transform(X_train)

print("done in %0.3fs" % (time() - t0))


clf = SVC(C=50, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))

print(clf.fit(X_train_pca, y_train).score(X_train_pca, y_train))

del X_train, X_train_pca, y_train, mris, mris_flat

# #############################################################################
# get test data
num = 110

mris_test = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)

path1 = np.chararray((num,), itemsize=12)

for j in range(num):
    path1[j], mris_test[j,:,:,:] = get_HGG_LGG_test(j)
   # print(path1[j])
   
        

n_samples_test, d, h, w, m = mris_test.shape
mris_test_flat = np.reshape(mris_test, (num, h*w*d*m))
X_test = mris_test_flat
print("Total dataset size:")
print("n_samples: %d" % n_samples_test)
print("Projecting the test data on the eigenmris orthonormal basis")
t0 = time()
X_test_pca = PCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
y_decis = clf.decision_function(X_test_pca)
print("done in %0.3fs" % (time() - t0))

labels_test = dict()

for p, result in enumerate(zip(y_pred, y_decis)):
    labels_test[path1[p]] = (result[0], result[1])
    print(path1[p], 'predicted grade:', result[0], 'decision_data:', result[1]) 

f = open("test_labels.pkl","wb")
pickle.dump(labels_test,f)
f.close()


# In[4]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.feature_selection import RFE
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274
numHGG = 220
mris = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)
labels = np.ndarray(shape=(num,), dtype = np.int16)
indices = np.random.permutation([i for i in range(labels.shape[0])])

for j in range(len(indices)):
   # print(j, indices[j])
    if indices[j] < numHGG: 
        mris[j,:,:,:] = get_HGG_train(indices[j])
        labels[j] = 0
    elif indices[j] >= numHGG: 
        mris[j,:,:,:] = get_LGG_train(indices[j]-numHGG)        
        labels[j] = 1    
    
        
# introspect the images arrays to find the shapes (for plotting)
n_samples, d, h, w, m = mris.shape
mris_flat = np.reshape(mris, (num, h*w*d*m))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

n_features = mris_flat.shape[1]


# the label to predict is the id of the person
label_names = ['HG', 'LG']
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    mris_flat, labels, test_size=0.1, random_state=48)
# #############################################################################
# Compute a PCA (eigenmris) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 25

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
PCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = PCA.components_.reshape((n_components, h*16, w*16))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = PCA.transform(X_train)
X_test_pca = PCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))




# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = {'C': [0.1, 1, 5, 10, 50, 100], 'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1],}

clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=10)

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                   param_grid, cv=5)                   param_grid, cv=5)

clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print(clf.fit(X_train_pca, y_train).score(X_train_pca, y_train))

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h*16, w*16)

# plot the gallery of the most significative eigenmris

eigenmri_titles = ["eigenmri %d" % i for i in range(eigenmris.shape[0])]
plot_gallery(eigenmris, eigenmri_titles, h*16, w*16)

plt.show()


# In[3]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.feature_selection import RFE
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274
numHGG = 220
mris = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)
labels = np.ndarray(shape=(num,), dtype = np.int16)
indices = np.random.permutation([i for i in range(labels.shape[0])])

for j in range(len(indices)):
   # print(j, indices[j])
    if indices[j] < numHGG: 
        mris[j,:,:,:] = get_HGG_train(indices[j])
        labels[j] = 0
    elif indices[j] >= numHGG: 
        mris[j,:,:,:] = get_LGG_train(indices[j]-numHGG)        
        labels[j] = 1    
    
        
# introspect the images arrays to find the shapes (for plotting)
n_samples, d, h, w, m = mris.shape
mris_flat = np.reshape(mris, (num, h*w*d*m))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

n_features = mris_flat.shape[1]


# the label to predict is the id of the person
label_names = ['HG', 'LG']
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    mris_flat, labels, test_size=0.1, random_state=48)
# #############################################################################
# Compute a PCA (eigenmris) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 25

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
PCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = PCA.components_.reshape((n_components, h*16, w*16))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = PCA.transform(X_train)
X_test_pca = PCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))




# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#param_grid = {'C': [1, 5, 10, 50], 'gamma': [0.001, 0.005, 0.01, 0.1],}
             

clf = SVC(C=1,class_weight='balanced', gamma=0.005)

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                   param_grid, cv=5)                   param_grid, cv=5)

clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)
print(clf.fit(X_train_pca, y_train).score(X_train_pca, y_train))

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h*16, w*16)

# plot the gallery of the most significative eigenmris

eigenmri_titles = ["eigenmri %d" % i for i in range(eigenmris.shape[0])]
plot_gallery(eigenmris, eigenmri_titles, h*16, w*16)

plt.show()


# In[ ]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274
numHGG = 220
mris = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)
labels = np.ndarray(shape=(num,), dtype = np.int16)
indices = np.random.permutation([i for i in range(labels.shape[0])])
for j in range(len(indices)):
   # print(j, indices[j])
    if indices[j] < numHGG: 
        mris[j,:,:,:] = get_HGG_train(indices[j])
        labels[j] = 0
    elif indices[j] >= numHGG: 
        mris[j,:,:,:] = get_LGG_train(indices[j]-numHGG)        
        labels[j] = 1
        
# introspect the images arrays to find the shapes (for plotting)
n_samples, d, h, w, m = mris.shape

mris_flat = np.reshape(mris, (num, h*w*d*m))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

#n_features = mris_flat.shape[1]
n_features = mris_flat.shape[1]/4

# the label to predict is the id of the person
label_names = ['HG', 'LG']
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    mris_flat, labels, test_size=0.25, random_state=40)


# #############################################################################
# Compute a PCA (eigenmris) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
PCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = PCA.components_.reshape((n_components, h*8, w*8))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = PCA.transform(X_train)
X_test_pca = PCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#param_grid = {'C': [5e1, 1e2, 5e2, 1e3, 5e3, 1e4], 'kernel': ['poly','rbf'], 'degree': [3,4,5]}
param_grid = {'n_neighbors': [3,4,5]}
clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                   param_grid, cv=5)                   param_grid, cv=5)

clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h*8, w*8)

# plot the gallery of the most significative eigenmris

eigenmri_titles = ["eigenmri %d" % i for i in range(eigenmris.shape[0])]
plot_gallery(eigenmris, eigenmri_titles, h*8, w*8)

plt.show()


# In[4]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274
numHGG = 220
mris = np.ndarray(shape=(num, d, h, w, m), dtype = np.float32)
labels = np.ndarray(shape=(num,), dtype = np.int16)
indices = np.random.permutation([i for i in range(labels.shape[0])])
for j in range(len(indices)):
   # print(j, indices[j])
    if indices[j] < numHGG: 
        mris[j,:,:,:] = get_HGG_train(indices[j])
        labels[j] = 0
    elif indices[j] >= numHGG: 
        mris[j,:,:,:] = get_LGG_train(indices[j]-numHGG)        
        labels[j] = 1
        
# introspect the images arrays to find the shapes (for plotting)
n_samples, d, h, w, m = mris.shape
mris_flat = np.reshape(mris, (num, h*w*d*m))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

n_features = mris_flat.shape[1]


# the label to predict is the id of the person
label_names = ['HG', 'LG']
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    mris_flat, labels, test_size=0.25, random_state=40)


# #############################################################################
# Compute a PCA (eigenmris) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 100

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
PCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = PCA.components_.reshape((n_components, h*8, w*8))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = PCA.transform(X_train)
X_test_pca = PCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#param_grid = {'C': [5e1, 1e2, 5e2, 1e3, 5e3, 1e4], 'kernel': ['poly','rbf'], 'degree': [3,4,5]}

estimator = SVC(C=50.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

#clf = GridSearchCV(SVC(gamma=0.1, class_weight='balanced'), param_grid, cv=5)

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                   param_grid, cv=5)                   param_grid, cv=5)
selector = RFE(estimator, 100, step=10)
selector = selector.fit(X_train_pca, y_train)
#clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
print("Best estimator found by feature eliminator:")
#print(selector.best_estimator_)
print(selector.ranking_)
print(selector.support_ )

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = selector.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h*8, w*8)

# plot the gallery of the most significative eigenmris

eigenmri_titles = ["eigenmri %d" % i for i in range(eigenmris.shape[0])]
plot_gallery(eigenmris, eigenmri_titles, h*8, w*8)

plt.show()


# In[4]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as pca
from sklearn.svm import SVC
from random import shuffle

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
num = 274

mris = np.array([fetch_mri(j) for j in indices])  
indices = np.random.permutation([i for i in range(l)])
                   
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w, d = mris.shape
mris_flat = np.reshape(mris, (num, h*w*d))    

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

n_features = mris_flat.shape[1]


# the label to predict is the id of the person
target_names = np.array(label_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    mris_flat, labels, test_size=0.25, random_state=42)


# #############################################################################
# Compute a PCA (eigenmris) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 100

print("Extracting the top %d eigenmris from %d mris"
      % (n_components, X_train.shape[0]))
t0 = time()
sparsePCA = pca(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

eigenmris = sparsePCA.components_.reshape((n_components, h*4, w*4))

print("Projecting the input data on the eigenmris orthonormal basis")
t0 = time()
X_train_pca = sparsePCA.transform(X_train)
X_test_pca = sparsePCA.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting mri's grades on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h*4, w*4)

# plot the gallery of the most significative eigenmris

eigenmri_titles = ["eigenmri %d" % i for i in range(eigenmris.shape[0])]
plot_gallery(eigenmris, eigenmri_titles, h*4, w*4)

plt.show()


# In[ ]:





# In[ ]:


#got messed up somehow

#use stacked numpy arrays from previous training and testing runs to get one patient at
# a time to run through u-net for BRATS2015
maskpath = "/home/julie/U-net_code_Tumor/brainmasks/"

def get_HGG_train(i):

    if i < 66:
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_train'][i:i+1].tolist()
           # print (path[0])
            
        with h5py.File('HGGstack_train1.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_train1'][155*i:155*(i+1),...]         
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_train'][155*i:155*(i+1),...]            
#             del hf

    if 66 <= i < 110:
        i1 = i - 66
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_valid'][i1:i1+1].tolist()
         #   print (path[0])
        with h5py.File('HGGstack_valid.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_valid'][155*i1:155*(i1+1),...]
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_valid'][155*i1:155*(i1+1),...]            
#             del hf

    if 110 <= i < 176:
        i2 = i - 44
        j = i - 110
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_train'][i2:i2+1].tolist()
         #   print (path[0])
        with h5py.File('HGGstack_train2.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_train2'][155*j:155*(j+1),...]         
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_train'][155*i2:155*(i2+1),...]            
#             del hf

    if 176 <= i < 220:
        i3 = i - 176
        with h5py.File('PathsHGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsHGG_"+ 'Flair' + '_test'][i3:i3+1].tolist()
        #    print (path[0])
        with h5py.File('HGGstack_test.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['HGGstack_test'][155*i3:155*(i3+1),...]
            del hf
#         with h5py.File('HGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['HGGground_test'][155*i3:155*(i3+1),...]            
#             del hf
            
 #   l1 = path[0].find('Flair.')
  #  l2 = path[0][l1+6:].find('Flair.')
   # print(l1,l2)
   # path1 = path[0][l1+l2+6:]
   # print(path1)
    p1 = path[0].find('pat')
    p2 = path[0].find('/VSD')
    
    brainmask = nib.load(os.path.join(maskpath+'HGG/'+path[0][p1:p2]+'/', 'brainmask.nii.gz')).get_data()
    brainmask = np.moveaxis(brainmask,-1,0)
    X_train = np.stack([X_train[:,:,:,i]*brainmask for i in range(4)], axis = -1)
  #  X_train = distort_imgs(X_train) 
   # if p == 1: X_train = np.stack([distort_imgs(X_train[n,:,:,:]) for n in range(X_train.shape[0])], axis=0)
    #return path, path1, ndimage.zoom(X_train, 0.1)
    return ndimage.zoom(X_train,(0.41, 0.2, 0.2, 1) )

def get_LGG_train(i,p=0):
    
    if i < 32:
        with h5py.File('LGGstack_train.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_train'][155*i:155*(i+1),...]         
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_train'][155*i:155*(i+1),...]            
#             del hf

    if 32 <= i < 43:
                i1 = i - 32
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            path[0][p1:p2])
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsLGG_"+ 'Flair' + '_train'][i:i+1].tolist()
         #   print (path[0])
            path = hf["PathsLGG_"+ 'Flair' + '_valid'][i1:i1+1].tolist()
           # print (path[0])
        with h5py.File('LGGstack_valid.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_valid'][155*i1:155*(i1+1),...]
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_valid'][155*i1:155*(i1+1),...]            
#             del hf
            
        
    if 43 <= i < 54:
        i3 = i - 43
        with h5py.File('PathsLGG_splits.h5', 'r') as hf:
            #  for key in hf:print(key, hf[key].shape)
            
            path = hf["PathsLGG_"+ 'Flair' + '_test'][i3:i3+1].tolist()
         #   print (path[0])
        with h5py.File('LGGstack_test.h5', 'r') as hf:
          #  for key in hf:print(key, hf[key].shape)
            X_train = hf['LGGstack_test'][155*i3:155*(i3+1),...]
            del hf
#         with h5py.File('LGGground_splits.h5', 'r') as hf:
#          #   for key in hf:print(key, hf[key].shape)        
#             y_train = hf['LGGground_test'][155*i3:155*(i3+1),...]            
#             del hf
  #  l1 = path[0].find('Flair.')
  #  l2 = path[0][l1+6:].find('Flair.')
   
  #  path1 = path[0][l1+l2+6:]
  #  print(path1)
    p1 = path[0].find('pat')
    p2 = path[0].find('/VSD')
   # print(path[0][p1:p2]) 
    brainmask = nib.load(os.path.join(maskpath+'LGG/'+path[0][p1:p2]+'/', 'brainmask.nii.gz')).get_data()
    brainmask = np.moveaxis(brainmask,-1,0)
    X_train = np.stack([X_train[:,:,:,i]*brainmask for i in range(4)], axis = -1)
   # print(X_train.shape)
    
  #  return path, path1, ndimage.zoom(X_train, 0.1)
    
   # if p == 1: X_train = np.stack([distort_imgs(X_train[n,:,:,:]) for n in range(X_train.shape[0])], axis=0)
  #  X_train = distort_imgs(X_train)   
   # print(X_train.shape)    
    
    return ndimage.zoom(X_train, (0.41, 0.2, 0.2, 1))

    
    
    
    
    
    
    
#path, path1, mri = get_HGG_train(0)
mri = get_LGG_train(0)
#print (path, path1, mri.shape)
print (mri.shape)
d, h, w, m = mri.shape

