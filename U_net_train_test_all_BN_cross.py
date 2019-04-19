#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os, time
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


# In[2]:


def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                                   padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                       padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out


# In[3]:


#augment data with horizontal and vertical flips, rotation, shift, shear, zoom, brightness, and elastic distortion
#image = mristack :flair, t1, t1c, t2, ground (shape = (240, 240 ))
def distort_imgs(image):
    x1, x2, x3, x4, y = image
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=0, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=1, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
   # x1, x2, x3, x4, y = tl.prepro.brightness_multi([x1, x2, x3, x4, y], gamma=0.2, gain=1, is_random=True)
    return x1, x2, x3, x4, y


# In[4]:


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


# In[5]:



def run_u_net(task='all'):
    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
  #  tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic 
    # Label 2: edema
    # label 3: non-enhancing tumor
    # Label 4: enhancing tumor
    
    import get_mixed_datasets as dataset
    X_train = dataset.X_train    
    y_train = dataset.y_train[:,:,:,np.newaxis]   
    X_test = dataset.X_test   
    y_test = dataset.y_test[:,:,:,np.newaxis]
        
    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'core':
        y_train = np.logical_and(y_train > 0, y_train != 2).astype(int)       
        y_test = np.logical_and(y_test > 0, y_test != 2).astype(int) 
    
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
 
        
    else:
        exit("Unknown task %s" % task)
###======================== HYPER-PARAMETERS ============================###
    batch_size = 15#10
    lr = 0.0001 
  #  lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 25#100
    print_freq_step = 100
    keep = 0.8
###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.338576 11.022792
    print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1

    showDataImages(X,4)
    showDataImages(y,1)
    nw, nh, nz = X.shape
   # vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis],
                X[:,:,2,np.newaxis], X[:,:,3,np.newaxis], y])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
      #  vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFIINE MODEL =======================###
            ## nz is 4 as we input all Flair, T1, T1c and T2.
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
            
            ## train inference
            net = u_net_bn(t_image, is_train=True, reuse=False, n_out=1)
            ## test inference
            net_test = u_net_bn(t_image, is_train=False, reuse=True, n_out=1)
            ###======================== DEFINE LOSS =========================###
            ## train losses
            out_seg = net.outputs
            dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
            dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
            ce_loss = tl.cost.binary_cross_entropy(out_seg, t_seg)
            loss = 0.5*(dice_loss+ce_loss)
            

            ## test losses
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])
        ###======================== DEFINE TRAIN OPTS =======================###
        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)
        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)
        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        ## update decay learning rate at the beginning of a epoch
       # if epoch !=0 and (epoch % decay_every == 0):
        #    new_lr_decay = lr_decay ** (epoch // decay_every)
         #   sess.run(tf.assign(lr_v, lr * new_lr_decay))
          #  log = " ** new learning rate: %f" % (lr * new_lr_decay)
           # print(log)
     #   elif epoch == 0:
       #     sess.run(tf.assign(lr_v, lr))
       #     log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
       #    print(log)

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels)],
                    fn=distort_imgs) # (10, 5, 240, 240, 1)
            b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,4,:,:,:]
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, nw, nh, nz)
            feed_dict = {t_image: b_images, t_seg: b_labels}
          #  feed_dict.update( net.all_drop ) 
            ## update network
            
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    feed_dict=feed_dict)
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            ## you can show the prediction here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            if _dice == 1: # DEBUG
                print("DEBUG")
                vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

#         ## save a prediction of training set
#         for i in range(batch_size):
#             if np.max(b_images[i]) > 0:
#                 vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
#                 break
#             elif i == batch_size-1:
#                 vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
#         ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
           # dp_dict = tl.utils.dict_to_one( net_test.all_drop )
            feed_dict = {t_image: b_images, t_seg: b_labels}
          #  feed_dict.update(dp_dict)
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    feed_dict=feed_dict)
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        print(" task: {}".format(task))
#         ## save a predition of test set
#         for i in range(batch_size):
#             if np.max(b_images[i]) > 0:
#                 vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
#                 break
#             elif i == batch_size-1:
#                 vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)


# In[6]:


#0 get_mixed_datasets, batch size 15, 25 epochs 
run_u_net()


# In[6]:


#1 get_mixed_datasets, batch size 15, 25 epochs 
run_u_net()


# In[6]:


#2 get_mixed_datasets, batch size 15, 25 epochs 
run_u_net()


# In[6]:


#3 get_mixed_datasets, batch size 15, 25 epochs 
run_u_net()


# In[6]:


#4 get_mixed_datasets, batch size 15, 25 epochs 
run_u_net()


# In[ ]:




