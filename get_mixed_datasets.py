import h5py
import numpy as np
#get datasets for training and testing with mixture of HGG and LGG
t = int(input('Enter t: '))


with h5py.File('HGGstack_train1.h5', 'r') as hf:
    #  for key in hf:print(key, hf[key].shape)
    if t == 0: X_train0 = hf['HGGstack_train1'][0::5,...]         
    elif t == 1: X_train0 = hf['HGGstack_train1'][1::5,...]
    elif t == 2: X_train0 = hf['HGGstack_train1'][2::5,...]
    elif t == 3: X_train0 = hf['HGGstack_train1'][3::5,...]
    elif t == 4: X_train0 = hf['HGGstack_train1'][4::5,...]
    del hf
with h5py.File('HGGstack_train2.h5', 'r') as hf:
    if t == 0: X_train1 = hf['HGGstack_train2'][0::5,...]         
    elif t == 1: X_train1 = hf['HGGstack_train2'][1::5,...]
    elif t == 2: X_train1 = hf['HGGstack_train2'][2::5,...]
    elif t == 3: X_train1 = hf['HGGstack_train2'][3::5,...]
    elif t == 4: X_train1 = hf['HGGstack_train2'][4::5,...]  
    del hf
with h5py.File('HGGstack_valid.h5', 'r') as hf:
    #   for key in hf:print(key, hf[key].shape)
    if t == 0: X_test0 = hf['HGGstack_valid'][0::5,...]
    elif t == 1: X_test0 = hf['HGGstack_valid'][1::5,...]
    elif t == 2: X_test0 = hf['HGGstack_valid'][2::5,...]
    elif t == 3: X_test0 = hf['HGGstack_valid'][3::5,...]
    elif t == 4: X_test0 = hf['HGGstack_valid'][4::5,...]               
    del hf       
with h5py.File('HGGstack_test.h5', 'r') as hf:
    if t == 0: X_test1 = hf['HGGstack_test'][0::5,...]   
    elif t == 1:  X_test1 = hf['HGGstack_test'][1::5,...]
    elif t == 2:  X_test1 = hf['HGGstack_test'][2::5,...]
    elif t == 3:  X_test1 = hf['HGGstack_test'][3::5,...]
    elif t == 4:  X_test1 = hf['HGGstack_test'][4::5,...]
    del hf
with h5py.File('HGGground_splits.h5', 'r') as hf:
    #   for key in hf:print(key, hf[key].shape)
    if t == 0:
        y_train0 = hf['HGGground_train'][0::5,...]
        y_test0 = hf['HGGground_valid'][0::5,...]
        y_test1 = hf['HGGground_test'][0::5,...] 
    elif t == 1:
        y_train0 = hf['HGGground_train'][1::5,...]
        y_test0 = hf['HGGground_valid'][1::5,...]
        y_test1 = hf['HGGground_test'][1::5,...]
    elif t == 2:
        y_train0 = hf['HGGground_train'][2::5,...]
        y_test0 = hf['HGGground_valid'][2::5,...]
        y_test1 = hf['HGGground_test'][2::5,...]
    elif t == 3:
        y_train0 = hf['HGGground_train'][3::5,...]
        y_test0 = hf['HGGground_valid'][3::5,...]
        y_test1 = hf['HGGground_test'][3::5,...]
    elif t == 4:
        y_train0 = hf['HGGground_train'][4::5,...]
        y_test0 = hf['HGGground_valid'][4::5,...]
        y_test1 = hf['HGGground_test'][4::5,...]          
    del hf
        
    
with h5py.File('LGGstack_train.h5', 'r') as hf:
    #   for key in hf:print(key, hf[key].shape)
    if t == 0: X_train2 = hf['LGGstack_train'][0::5,...]
    elif t == 1: X_train2 = hf['LGGstack_train'][1::5,...]
    elif t == 2: X_train2 = hf['LGGstack_train'][2::5,...]
    elif t == 3: X_train2 = hf['LGGstack_train'][3::5,...]
    elif t == 4: X_train2 = hf['LGGstack_train'][4::5,...]
    del hf
with h5py.File('LGGstack_valid.h5', 'r') as hf:
    #   for key in hf:print(key, hf[key].shape)
    if t == 0: X_test2 = hf['LGGstack_valid'][0::5,...]
    elif t == 1: X_test2 = hf['LGGstack_valid'][1::5,...]
    elif t == 2: X_test2 = hf['LGGstack_valid'][2::5,...]
    elif t == 3: X_test2 = hf['LGGstack_valid'][3::5,...]
    elif t == 4: X_test2 = hf['LGGstack_valid'][4::5,...]
    del hf
with h5py.File('LGGstack_test.h5', 'r') as hf:
    #for key in hf:print(key, hf[key].shape)
    if t == 0: X_test3 = hf['LGGstack_test'][0::5,...]
    elif t == 1: X_test3 = hf['LGGstack_test'][1::5,...]
    elif t == 2: X_test3 = hf['LGGstack_test'][2::5,...]
    elif t == 3: X_test3 = hf['LGGstack_test'][3::5,...]
    elif t == 4: X_test3 = hf['LGGstack_test'][4::5,...]
    del hf
    
with h5py.File('LGGground_splits.h5', 'r') as hf:
    #   for key in hf:print(key, hf[key].shape)
    if t == 0:
        y_train1 = hf['LGGground_train'][0::5,...]
        y_test2 = hf['LGGground_valid'][0::5,...]
        y_test3 = hf['LGGground_test'][0::5,...] 
    elif t == 1:
        y_train1 = hf['LGGground_train'][1::5,...]
        y_test2 = hf['LGGground_valid'][1::5,...]
        y_test3 = hf['LGGground_test'][1::5,...]
    elif t == 2:
        y_train1 = hf['LGGground_train'][2::5,...]
        y_test2 = hf['LGGground_valid'][2::5,...]
        y_test3 = hf['LGGground_test'][2::5,...]
    elif t == 3:
        y_train1 = hf['LGGground_train'][3::5,...]
        y_test2 = hf['LGGground_valid'][2::5,...]
        y_test3 = hf['LGGground_test'][2::5,...]
    elif t == 4:
        y_train1 = hf['LGGground_train'][4::5,...]
        y_test2 = hf['LGGground_valid'][4::5,...]
        y_test3 = hf['LGGground_test'][4::5,...]          
    del hf
        
        
X_train = np.concatenate([X_train0,X_train1,X_train2])
print ('X_train:', X_train.shape)
X_test = np.concatenate([X_test0,X_test1,X_test2,X_test3])
print ('X_test:', X_test.shape)
y_train = np.concatenate([y_train0,y_train1])
print ('y_train:', y_train.shape)
y_test = np.concatenate([y_test0,y_test1,y_test2,y_test3])
print ('y_test:', y_test.shape)
    

    
