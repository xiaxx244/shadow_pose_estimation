import sys
import os
import glob
import natsort
import numpy as np
import scipy
import keras
import utls
import cv2
import time
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
import tensorflow as tf

from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

from data_load import Dataloader
import Network
import yaml
resnet50 = Network.build_resnet50((256,256,3))
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


def build_test_model():
    img_rows = 256
    img_cols = 256
    img_channels = 3
    model = Network.build_imh((img_rows, img_cols, img_channels))
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

    # Compile model
    print("[INFO] compiling model...")
    opt = Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights("models3/200_.h5", by_name=True)
    model.trainable = False
    model.compile(loss=losses,
                metrics=[utls.bright_mae,
                        utls.bright_mse, utls.bright_SSIM],
                optimizer=opt)

    model.summary()
    return model

def transform_img(img, height, width):
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128
    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / height
        new_width = int(img.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(img, (new_width, height))
        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / width
        new_height = int(img.shape[0] / scale)
        diff = (height - new_height) // 2
        img = cv2.resize(img, (width, new_height))
        im[diff:diff + new_height, :, :] = img
    im = np.float32(im) / 127.5 - 1
    return im

def rescale_to_image(img):
    # Scale a numpy array in np.float32 format to cv image format uint8
    # img = np.load(path)
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for c in range(3):
        imin = img[:,:,c].min()
        imax = img[:,:,c].max()
        scale = 255 / (imax - imin)
        offset = 255 - scale * imax
        new_img[:,:,c] = (scale * img[:,:,c] + offset).astype(np.uint8)
    return new_img

def test():
    # Build Model
    model = build_test_model()
    # num_val=1024
    # num_samples = len(im_path)-num_val
    # num_samples = 40000
    # train_list=im_path[:num_samples]
    img_rows = 256
    img_cols = 256
    img_channels = 3
    input_folder = "../../../../media/bizon/Elements/shadow_test/private/l_far_male_filter3_shadow"
    path = glob.glob(input_folder+'/*.*')

    for i in range(len(path)):
        img_path = path[i]
        img = cv2.imread(img_path)
        img = transform_img(img, 256, 256) # Convert to (0,1)
        # Check if transform has error
        img = img[np.newaxis, :]
        start_time = time.clock()
        [_,_,out_pred] = model.predict(img) # have 1 output images
        end_time = time.clock()
        print('The' + str(i + 1) + 'th image\'s Time:' + str(end_time - start_time))

        filename = os.path.basename(img_path).split('.')[0]
        output = out_pred[0].reshape(img_rows, img_cols, img_channels)
        result_img = rescale_to_image(output)
        save_path = "l_far_male_filter3" + "/" + filename + "_enhanced.jpg"
        cv2.imwrite(save_path, result_img)


if __name__ == "__main__":
    test()
