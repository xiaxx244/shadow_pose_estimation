import os
import pickle
import glob
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utls
import natsort
import tensorflow as tf
import keras
from data_load import Dataloader

from keras.applications.vgg19 import VGG19
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.models import Model

from tensorflow.python.client import device_lib

def get_data_loader():
    #  Dataloader
    img_rows = 256
    img_cols = 256
    img_channels = 3
    image_path_name = 'segmentation'
    data_loader = Dataloader(dataset_name=image_path_name,
                            crop_shape=(img_rows, img_cols))
    return data_loader

def get_transform_image(path, data_loader):
    img = cv2.imread(path)  # b, g, r format
    img = data_loader.transform_img(img, 256, 256) # convert to (0, 1)
    return img

def ssim_test(num_data):
    train_img_path = glob.glob("/home/yifan/Documents/ShadowSense_code/image_enhancement/results/pretrain_testing/2_residual_db/train_on_hazy/*.jpg")
    data_loader = get_data_loader()
    #  Calculate SSIM
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))

    with open(os.getcwd() + "ssim.txt", 'w') as f_handle:
        print("1")
        f_handle.write("image name\tPre-train on Hazy\tTrain on blurry\n")
        for i in range(len(train_img_path)):
            if "blurry" in train_img_path[i].split("/")[-1]:
                continue
            hazy_img = get_transform_image(train_img_path[i], data_loader)
            hazy_img = hazy_img[np.newaxis,:]

            blur_img_name = train_img_path[i].split("/")[-1].split("_")[0] + "_blurry.jpg"
            blur_img_path = os.path.dirname(train_img_path[i]) + "/" + blur_img_name
            blur_img = get_transform_image(blur_img_path, data_loader)
            blur_img = blur_img[np.newaxis, :]

            shadow_img_name = train_img_path[i].split("/")[-1].split("_")[0] + ".jpg"
            shadow_path = "/home/yifan/Documents/ShadowSense_code/image_enhancement/q_t/" + shadow_img_name
            shadow_img = get_transform_image(blur_img_path, data_loader)
            shadow_img = shadow_img[np.newaxis, :]

            ssim_before = utls.bright_SSIM(shadow_img, hazy_img)
            ssim_after = utls.bright_SSIM(shadow_img, blur_img)
            ssim_before = sess.run(ssim_before)
            ssim_after = sess.run(ssim_after)
            f_handle.write("img: {}, ssim: {}-> {} \n".format(train_img_path[i].split("/")[-1].split("_")[0], ssim_before, ssim_after))

        # sum_ssim_original = sum_ssim_original / cnt
        # sum_ssim_result = sum_ssim_result / cnt
        # f_handle.write("Total average ssim : {} ---> {}, cnt: {} \n".format(sum_ssim_original, sum_ssim_result, cnt))

def build_vgg19(image_input):
    image_input=Input(shape=image_input)
    with tf.device('/gpu:0'):
        model = VGG19(input_tensor=image_input,include_top=False, weights='imagenet')
    x = model.get_layer('block3_conv4').output
    #x = Conv2D(16, (3,3), padding="same", activation="relu")(x)
    model.trainable = False
    with tf.device('/gpu:0'):
        model=Model(inputs=model.input, outputs=x)
    return model

"""
def perceptual_loss_test():
    #  Dataloader
    train_img_path = natsort.natsorted(
        glob.glob(os.getcwd() + '/data/images/*'), reverse=False)
    
    data_loader = get_data_loader()

    # build vgg model
    crop_shape = (256, 256, 3)
    # vgg_model = build_vgg19(crop_shape)
    # vgg_model.trainable = False

    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
    cnt = 0
    vgg_loss_sum_prev = 0
    vgg_loss_sum_after = 0 
    with open("vgg_loss.txt", 'w') as f_handle:
        f_handle.write("name         original         result\n")
        for i in range(len(train_img_path)):
            #### Get train image
            train_image = get_transform_image(train_img_path[i], data_loader)

            #### Get real image
            real_image_path = os.getcwd() + "/data/real_images/" + train_img_path[i].split("/")[-1].split("_")[0]+".jpg"
            real_image = get_transform_image(real_image_path, data_loader)
            
            #### Get result image
            result_image_path = os.getcwd() + "/data/results/model_5_4/imgs/" + "stage3_" + train_img_path[i].split("/")[-1]
            result_image = get_transform_image(result_image_path, data_loader)
                        
            # ssim
            train_image = train_image[np.newaxis,:]
            real_image = real_image[np.newaxis,:]
            result_image = result_image[np.newaxis, :]

            # calculate average l2-norm of (map_y_true - map_y_pred)
            map_y_true = vgg_model.predict(train_image) 
            map_y_pred = vgg_model.predict(real_image)
            vgg_loss_train_real = keras.backend.mean(keras.backend.square(map_y_true - map_y_pred))
            
            map_y_true = vgg_model.predict(result_image)
            map_y_pred = vgg_model.predict(real_image)
            vgg_loss_result_real = keras.backend.mean(keras.backend.square(map_y_true-map_y_pred))

            vgg_loss_train_real = sess.run(vgg_loss_train_real)
            vgg_loss_result_real = sess.run(vgg_loss_result_real)
            file_name = "stage3_" + train_img_path[i].split("/")[-1]
            f_handle.write("{}         {}         {} \n".format(file_name, vgg_loss_train_real, vgg_loss_result_real))

            cnt = cnt + 1
            vgg_loss_sum_prev = vgg_loss_sum_prev + vgg_loss_train_real
            vgg_loss_sum_after = vgg_loss_sum_after + vgg_loss_result_real
        
        vgg_loss_sum_prev = vgg_loss_sum_prev / cnt
        vgg_loss_sum_after = vgg_loss_sum_after / cnt
        f_handle.write("vgg average loss: {} -> {} \n".format(vgg_loss_sum_prev, vgg_loss_sum_after))
"""

def print_history(path):
    with open(path, 'rb') as handle:
        history = pickle.load(handle)
    print(history.keys())
    
    # loss
    if 'loss' in history.keys() and 'val_loss' in history.keys():
        plt.plot(history['loss'], marker='.')
        plt.plot(history['val_loss'])
        plt.legend(['loss', 'validation loss'], loc = 'upper right')
        plt.title('Overall Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.dirname(path) + '/loss.png')
        plt.close()

    # ssim
    plt.plot(history['bright_SSIM_sigma'], marker='.')
    plt.plot(history['val_bright_SSIM_sigma'], marker='.')
    plt.title('ssim')
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.grid()
    plt.legend(['ssim', 'val_ssim'], loc='lower right')
    plt.savefig(os.path.dirname(path) + "/ssim.png")
    plt.close()

    # MAE
    plt.plot(history['bright_mae'], marker='.')
    plt.plot(history['val_bright_mae'], marker='.')
    plt.title('MAE')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.grid()
    plt.legend(['bright_mae', 'val_bright_mae'], loc='upper right')
    plt.savefig(os.path.dirname(path) + "/mae.png")
    plt.close()

    # MSE
    plt.plot(history['bright_mse'], marker='.')
    plt.plot(history['val_bright_mse'], marker='.')
    plt.title('MSE')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.grid()
    plt.legend(['bright_mse', 'val_bright_mse'], loc='upper right')
    plt.savefig(os.path.dirname(path) + "/mse.png")
    plt.close()

    # lr
    plt.plot(history['lr'], marker='.')
    plt.title('learning-rate')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.grid()
    plt.savefig(os.path.dirname(path) + '/Learning_Rate.png')
    plt.close()

    plt.plot(history['bright_psnr'], marker='.')
    plt.plot(history['val_bright_psnr'], marker='.')
    plt.legend(['bright_psnr', 'val_bright_psnr'], loc='lower right')
    plt.title('PSNR')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid()
    plt.savefig(os.path.dirname(path) + '/PSNR.png')
    plt.close()
    print("done")

def save_history(history, result_dir, prefix):
    with open(os.path.join(result_dir, '{}_result.pickle'.format(prefix)), 'wb') as fp:
        pickle.dump(history.history, fp)

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

def transform_numpy_to_image():
    # Post process numpy arrays, save images
    train_img_path = natsort.natsorted(
        glob.glob(os.getcwd() + '/data/results/single_uwcnn_small_sigma/*.npy'), reverse=False)

    for i in range(len(train_img_path)):
        # convert to image
        result_img = rescale_to_image(train_img_path[i])
        file_name = os.path.basename(train_img_path[i]).split('.')[0] + ".jpg"
        save_path = os.getcwd() + '/data/results/single_uwcnn_small_sigma/imgs/' + file_name
        cv2.imwrite(save_path, result_img)
        print('save i : {}, {}'.format(i, file_name))

if __name__ == "__main__":
    path = "/home/tyf/Documents/image_enhancement/results/0807_yifan_RDN5_3RB/RDN5_3_residual_blocks_weights_result.pickle"
    print_history(path)