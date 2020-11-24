from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Conv2DTranspose, Reshape, Dropout, concatenate, Concatenate, multiply, add, MaxPooling2D, Lambda, Activation, subtract, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
import imageio
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import Model
from data_load import Dataloader
from keras.utils import plot_model
from scipy import misc
from glob import glob
import tensorflow as tf
import numpy as np
import scipy
import platform
import keras
import os
import random
import Network
import utls
import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join('../', 'models/')))
import natsort
K.clear_session()

img_rows = 256
img_cols = 256
img_channels = 3
crop_shape = (img_rows, img_cols, img_channels)
input_shape = (img_rows, img_cols, img_channels)
data_loader = Dataloader(dataset_name=image_path, crop_shape=(img_rows, img_cols))
resnet50 = Network.build_resnet50(crop_shape)
resnet50.trainable = False

def my_loss(y_true, y_pred):
    MSE_loss = K.mean(K.abs(y_pred[:,:,:,:3] - y_true))
    SSIM_loss = utls.tf_ssim(tf.expand_dims(y_pred[:, :, :, 0], -1),tf.expand_dims(y_true[:, :, :, 0], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 1], -1), tf.expand_dims(y_true[:, :, :, 1], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 2], -1), tf.expand_dims(y_true[:, :, :, 2], -1))
    #psnr=tf.clip_by_norm(tf.image.psnr(y_pred[:,:,:,:3], y_true,max_val=1.0),1)
    return  3-SSIM_loss+3*MSE_loss

def my_loss1(y_true,y_pred):
    #distortion loss calculated using iou
    fake_features = resnet50(y_pred)
    real_features = resnet50(y_true)
    resnet_loss = 2*K.mean(K.abs(fake_features-real_features))
    loss = my_loss(y_true,y_pred)+resnet_loss
    return loss

def my_loss3(y_true, y_pred):
    MSE_loss = K.mean(K.square(y_pred[:,:,:,:3] - y_true))
    SSIM_loss = utls.tf_ssim(tf.expand_dims(y_pred[:, :, :, 0], -1),tf.expand_dims(y_true[:, :, :, 0], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 1], -1), tf.expand_dims(y_true[:, :, :, 1], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 2], -1), tf.expand_dims(y_true[:, :, :, 2], -1))
    #distortion loss calculated using iou
    #resnet_loss = K.mean(K.abs(y_pred[:, :, :, 3:19] - y_pred[:, :, :, 19:35]))
    edge_map1=tf.image.sobel_edges(y_pred[:,:,:,:3])
    edge_map2=tf.image.sobel_edges(y_true)
    ssim1 = K.mean(tf.image.ssim(edge_map1, edge_map2, max_val=255, filter_size=3,
                          filter_sigma=1.5, k1=0.01, k2=0.03))
    dist_loss=1-ssim1+K.mean(K.square(edge_map1-edge_map2))+1-tf.clip_by_norm(tf.image.psnr(y_pred[:,:,:,:3], y_true,max_val=1.0),1)

    loss = 3-SSIM_loss+MSE_loss + dist_loss
    return loss

if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.isdir('./models'):
    os.makedirs('./models')
if not os.path.isdir('./results'):
    os.makedirs('./results')


def f1(x):
    return x[:, :, :, :3]

def f2(x):
    return x[:, :, :, 3:]

def f3(x):
    y=tf.image.resize(x,tf.constant([256,256]))
    return tf.reshape(y,[-1, 256, 256, 16])

def f4(x):
    return tf.reshape(x,[-1, 256, 256, 2])

# Build the network
imh = Network.build_imh(crop_shape)
#img_B = Lambda(f2)(inputs)
losses = {
        "conv2d_9": my_loss,
        "conv2d_18": my_loss1,
        "conv2d_27": my_loss3,
        }
lossWeights = {"conv2d_9": 1.0,
               "conv2d_18": 1.0,
               "conv2d_27": 1.0,
              }

opt = Adam(lr=1e-04, beta_1=0.90, beta_2=0.999, epsilon=1e-08)

imh.compile(loss=losses, loss_weights=lossWeights,
                        metrics=[utls.bright_mae,
                           utls.bright_mse, utls.bright_SSIM],
                          optimizer=opt)


'''
if os.path.exists('models3/139_.h5'):
    imh.load_weights('models3/139_.h5')
    imh.summary()
'''

imh.summary()
#[fake_B,fake_B2,fake_B3] = imh(img_A)

def scheduler(epoch):
    lr = K.eval(imh.optimizer.lr)
    print("LR =", lr)
    #if lr>=1e-4:
    lr = lr * 0.999
    return lr
class Show_History(keras.callbacks.Callback):
    def on_epoch_end(self, val_loss=None, logs=None):
        # save model
        global num_epoch
        global im_path
        global img_rows
        global img_cols
        global imh
        global test
        global data_loader
        num_epoch += 1
        modelname = './models3/'+str(num_epoch) + '_.h5'
        imh.save_weights(modelname)

        # test val data
        test_list=im_path[89952:99968]
        number = 0
        psnr_ave = 0

        for i in range(len(test_list)):
            img_A_path=test_list[i]
            img_B = cv2.imread("../../../../media/bizon/Elements/OTS/train/OTSclear/"+img_A_path.split("_")[0].split("/")[-1]+".png")
            img_A = cv2.imread(img_A_path)
            #cv2.imwrite("val_real_images/"+img_A_path.split("_")[0].split("/")[-1]+".jpg",img_B)




            crop_img_A = data_loader.transform_img(img_A,256,256)
            #print(crop_img_A.shape)
            crop_img_B = data_loader.transform_img(img_B,256,256)

            [fake_B,fake_B2,fake_B3] = imh.predict(crop_img_A.reshape(1,256,256,3))
            [identy_B,identy_B2,identy_B3] = imh.predict(crop_img_B.reshape(1,256,256,3))

            #out_img = np.concatenate([crop_img_A.reshape(1,256,256,3), fake_B, crop_img_B.reshape(1,256,256,3), identy_B], axis=2)
            #out_img = out_img[0, :, :, :]

            fake_B3 = fake_B3[0, :, :, :]
            #img_B = crop_img_B[0, :, :, :]

            clean_psnr = utls.psnr_cau(fake_B3, crop_img_B)
            L_psnr = ("%.4f" % clean_psnr)

            number += 1
            psnr_ave += clean_psnr

            filename = os.path.basename(test_list[i])
            img_name = 'val_images7/' + str(num_epoch) + '_' + L_psnr + '_' + filename
            utls.imwrite(img_name, fake_B3)
        psnr_ave /= number
        print('------------------------------------------------')
        print("[Epoch %d]  [PSNR_AVE :%f]" % (num_epoch,  psnr_ave))
        print('------------------------------------------------')

show_history = Show_History()
#save_file = './models/imh_weights.h5'
num_epoch = 0
#checkpoint = ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, period=1)
change_lr = LearningRateScheduler(scheduler)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                         embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
nanstop = keras.callbacks.TerminateOnNaN()
reducelearate = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, min_lr=1e-10)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=3, patience=0, verbose=0, mode='min')

im_path=natsort.natsorted(glob("../../../../media/bizon/Elements/OTS/train/OTShaze/*"),reverse=False)
train_list=im_path[:69952]
val_list=im_path[69952:89952]
train=data_loader.load_data(fake_list=train_list)
val=data_loader.load_data(fake_list=val_list)
train_batch_size = 64
val_batch_size= 16

history=imh.fit_generator(
        train,
        steps_per_epoch=69952 // train_batch_size,
        epochs=200,
        validation_data=val,
        validation_steps=20000 // val_batch_size,
        callbacks=[show_history, tbCallBack,  change_lr, nanstop, reducelearate])

utls.plot_history(history, './results/', 'imh')
#utls.save_history(history, './results/', 'imh')

print('Done!')
