import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy
import os
import cv2 as cv
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'models')))

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir, prefix):

    plt.plot(history.history['loss'], marker='.')
    #plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(prefix)))
    plt.close()


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    #acc = history.history['acc']
    val_loss = history.history['val_loss']
    #val_acc = history.history['val_acc']
    #nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tval_loss\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\n'.format(
                i, loss[i], val_loss[i]))
        fp.close()

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def bright_mae(y_true, y_pred):
    return K.mean(K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3]))

def bright_mse(y_true, y_pred):
    return K.mean((y_pred[:,:,:,:3] - y_true[:,:,:,:3])**2)

def bright_AB(y_true, y_pred):
            return K.abs(K.mean(y_true[:,:,:,:3])-K.mean(y_pred[:,:,:,:3]))

def log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def bright_psnr(y_true, y_pred):
    mse = K.mean((K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3])) ** 2)
    max_num = 1.0
    psnr = 10 * log10(max_num ** 2 / mse)
    return psnr

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2)),
                        (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                    (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def bright_SSIM(y_true, y_pred):
    SSIM_loss = tf_ssim(tf.expand_dims(y_pred[:,:,:,0], -1), tf.expand_dims(y_true[:,:,:,0], -1))+tf_ssim(tf.expand_dims(y_pred[:,:,:,1], -1), tf.expand_dims(y_true[:,:,:,1], -1)) + tf_ssim(tf.expand_dims(y_pred[:,:,:,2], -1), tf.expand_dims(y_true[:,:,:,2], -1))
    return SSIM_loss/3

def psnr_cau(y_true, y_pred):
    mse = np.mean((np.abs(y_pred - y_true)) ** 2)
    max_num = 1.0
    psnr = 10 * np.log10(max_num ** 2 / mse)
    return psnr

def save_model(model, name, epoch, batch_i):
    modelname = './Res_models/' + str(epoch) + '_' + str(batch_i) + name + '.h5'
    model.save_weights(modelname)

def imread_color(path):
    img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH) / 255.
    b, g, r = cv.split(img)
    img_rgb = cv.merge([r, g, b])
    return img_rgb
    # return scipy.misc.imread(path, mode='RGB').astype(np.float) / 255.

def imwrite(path, img):
    r, g, b = cv.split(img*255)
    img_rgb = cv.merge([b, g, r])
    cv.imwrite(path, img_rgb)
    # scipy.misc.toimage(img * 255, high=255, low=0, cmin=0, cmax=255).save(path)

def range_scale(x):
    return x * 2 - 1.
