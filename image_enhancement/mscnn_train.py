import sys
import os
sys.path.append(os.path.abspath(os.path.join('../image_enhancement')))

import cv2
import natsort
import glob
import datetime

import tensorflow as tf
#from tensorflow import keras
import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, TensorBoard
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.applications.vgg19 import VGG19
from data_load import Dataloader
import utls
import mscnn


def build_vgg19(image_input):
    image_input=Input(shape=image_input)
    with tf.device('/cpu:0'):
        model = VGG19(input_tensor=image_input,include_top=False, weights='imagenet')
    x = model.get_layer('block3_conv4').output
    #x = Conv2D(16, (3,3), padding="same", activation="relu")(x)
    model.trainable = False
    with tf.device('/cpu:0'):
        model=Model(inputs=model.input, outputs=x)
    return multi_gpu_model(model,gpus=2)

img_rows = 256
img_cols = 256
img_channels = 3
crop_shape = (img_rows, img_cols, img_channels)
input_shape = (img_rows, img_cols, img_channels)
image_path = 'segmentation'
data_loader = Dataloader(dataset_name=image_path, crop_shape=(img_rows, img_cols))
m = build_vgg19(crop_shape)
m.trainable = False


# Development LOG
# TODO : 1. Test the MSCNN architecture (DONE)
#        2. Test all loss functions
#        3. Write the training code


# TODO how to incorporate regional loss, this lead to green color
def regional_loss(y_true, y_pred, percent):
    percent = 0.4
    index = int(256 * 256 * percent - 1)
    gray1 = 0.39 * y_pred[:, :, :, 0] + 0.5 * \
        y_pred[:, :, :, 1] + 0.11 * y_pred[:, :, :, 2]
    gray = tf.reshape(gray1, [-1, 256 * 256])
    gray_sort = tf.nn.top_k(-gray, 256 * 256)[0]
    yu = gray_sort[:, index]
    yu = tf.expand_dims(tf.expand_dims(yu, -1), -1)
    mask = tf.to_float(gray1 <= yu)
    mask1 = tf.expand_dims(mask, -1)
    mask = tf.concat([mask1, mask1, mask1], -1)

    low_fake_clean = tf.multiply(mask, y_pred[:, :, :, :3])
    high_fake_clean = tf.multiply(1 - mask, y_pred[:, :, :, :3])
    low_clean = tf.multiply(mask, y_true[:, :, :, :])
    high_clean = tf.multiply(1 - mask, y_true[:, :, :, :])
    Region_loss = K.mean(K.abs(low_fake_clean - low_clean)
                         * 4 + K.abs(high_fake_clean - high_clean))

    return Region_loss


def ssim_loss(y_true, y_pred):
    loss = 1 - tf.image.ssim(y_true, y_pred, max_val=255,
                             filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return loss


def mse_loss(y_true, y_pred):
    mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
    return mse_loss


def vgg_loss(y_true, y_pred):
    # construct a vgg19 network
    # calculate feature map for y_true and y_pred
    map_y_true = m(y_true)
    map_y_pred = m(y_pred)

    # calculate average l2-norm of (map_y_true - map_y_pred)
    loss = K.mean(K.square(map_y_true-map_y_pred))
    return loss

# distortion loss using edge map (TODO)


def distortion_loss(y_true, y_pred):
    edge_map1 = tf.image.sobel_edges(y_pred)
    edge_map2 = tf.image.sobel_edges(y_true)
    ssim1 = K.mean(tf.image.ssim(edge_map1, edge_map2, max_val=255, filter_size=3,
                                 filter_sigma=1.5, k1=0.01, k2=0.03))
    dist_loss = 3-ssim1+K.mean(K.square(edge_map1 - edge_map2))
    return dist_loss


def stage1_loss(y_true, y_pred):
    """
    The loss function for MSCNN, stage1
    SSIM + Regional Loss + MSE
    """
    loss_ssim = ssim_loss(y_true, y_pred)
    loss_mse = mse_loss(y_true, y_pred)
    loss_region = regional_loss(y_true, y_pred, 0.4)
    loss = loss_ssim + loss_mse + loss_region
    return loss


def stage2_loss(y_true, y_pred):
    """
    The loss function for MSCNN, stage2
    SSIM + MSE
    """
    loss_ssim = ssim_loss(y_true, y_pred)
    loss_mse = mse_loss(y_true, y_pred)
    loss = loss_ssim + loss_mse
    return loss


def stage3_loss(y_true, y_pred):
    """
    The loss function for MSCNN, stage3
    SSIM + MSE + perceptual + distortion loss
    """
    loss_ssim = ssim_loss(y_true, y_pred)
    loss_mse = mse_loss(y_true, y_pred)
    loss_perc = vgg_loss(y_true, y_pred)
    loss_dist = distortion_loss(y_true, y_pred)

    loss = loss_ssim + loss_mse + loss_perc + loss_dist
    return loss


if __name__ == "__main__":
    # Initialize the MSCNN model
    mscnn_object = mscnn.MSCNN(
        num_res_blocks=10, kernel_size=5, nb_channels_in=64)

    # Initialize the loss functions
    losses = {
        "conv2d_22": stage1_loss,
        "conv2d_44": stage2_loss,
        "conv2d_66": stage3_loss
    }
    lossWeights = {"conv2d_22": 1.0,
                   "conv2d_44": 1.0, "conv2d_66": 1.0}

    # Compile model
    print("[INFO] compiling model...")
    opt = Adam(lr=1*1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = mscnn_object.build_model(
            input_height=256, input_width=256, nChannels=3)

    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss=losses, loss_weights=lossWeights,
                  metrics=[utls.bright_mae,
                           utls.bright_mse, utls.bright_SSIM],
                  optimizer=opt)
    model.summary()

    # Separate training set and validation set
    im_path = natsort.natsorted(
        glob.glob("../segmentation/images/*"), reverse=False)
    # train_list = im_path[:60816]
    # val_list = im_path[60816:]

    train_list = im_path[:816]
    val_list = im_path[60816:60838]
    # load data
    # Define dataloader
    img_rows = 256
    img_cols = 256
    img_channels = 3
    image_path = 'segmentation'
    data_loader = Dataloader(dataset_name=image_path,
                             crop_shape=(img_rows, img_cols))

    train = data_loader.load_data(fake_list=train_list)
    val = data_loader.load_data(fake_list=val_list)

    # Define training tools
    save_file = './models3/mscnn_weights.h5'
    checkpoint = ModelCheckpoint(save_file, monitor='val_loss',
                                 save_best_only=True, save_weights_only=True, verbose=1, period=1)

    def scheduler(epoch):
        lr = K.eval(model.optimizer.lr)
        print("LR =", lr)
        lr = lr * 0.99  # decrease the learning rate
        return lr
    change_lr = LearningRateScheduler(scheduler)
    #log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
                            # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    nanstop = TerminateOnNaN()

    # reduce learning rate
    reducelearate = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=2, min_lr=1e-10)
    earlystop = EarlyStopping(
        monitor='loss', min_delta=3, patience=0, verbose=0, mode='min')

    # Training
    train_batch_size = 16
    val_batch_size = 16
    # history = model.fit_generator(
    #     train,
    #     steps_per_epoch=60816 // train_batch_size,
    #     epochs=200,
    #     validation_data=val,
    #     validation_steps=10000 // val_batch_size,
    #     callbacks=[tbCallBack, checkpoint, change_lr, nanstop, reducelearate])

    history=model.fit_generator(
        train,
        epochs=50,
        steps_per_epoch = 60816 // train_batch_size,
        callbacks=[checkpoint, change_lr, nanstop, reducelearate],
        validation_data=val,
        validation_steps= 32 // val_batch_size,)

    utls.plot_history(history, './results1/', 'mscnn')
    #utls.save_history(history, './results/', 'mscnn')

    print('Done!')
