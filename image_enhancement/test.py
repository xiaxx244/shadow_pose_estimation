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

from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

import mscnn
from data_load import Dataloader
import post_process
import dense_net as RDN
import loss_utils
import Network
import yaml


def build_test_model(model_name, model_weight_path):
    img_rows = 256
    img_cols = 256
    img_channels = 3
    if model_name == 'yifan':
        model = RDN.build_test_model((img_rows, img_cols, img_channels), 3, 16)
        losses = {"add_17": loss_utils.my_loss2}
    else:
        model = Network.build_imh((img_rows, img_cols, img_channels))
        losses = {
                "conv2d_9": Network.my_loss1,
                "conv2d_17": Network.my_loss3,
                }
        lossWeights = {"conv2d_9": 1.0,
                    "conv2d_17": 1.0,
                    }

    # Compile model
    print("[INFO] compiling model...")
    opt = Adam(lr=5 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(model_weight_path, by_name=True)
    model.trainable = False
    model.compile(loss=losses,
                metrics=[utls.bright_mae,
                        utls.bright_mse, utls.bright_SSIM],
                optimizer=opt)

    model.summary()
    return model

def test():
    with open("/home/tyf/Documents/image_enhancement/config/test_config.yaml", 'r') as f:
        config_file = f.read()
        config_dict = yaml.load(config_file)  # 用load方法转字典
 
    # Build Model
    model = build_test_model(config_dict['model'], config_dict['model_weight_path'])

    action_type = ['clap_hands', 'cross_arm', 'hands_on_waist', 'jump', 'kick', 'lunge', 'squat', 'stretch', 'wave']
    num_each_type = 500
    num_test = num_each_type * len(action_type)
    base_path = config_dict['data_path']
    path = []
    for action in action_type:
        cur_path =  natsort.natsorted(glob.glob(base_path + action + '/*/shadow/*.jpg'), reverse=False)
        path.extend(cur_path[len(cur_path)-num_each_type:len(cur_path)])

    # num_val=1024
    # num_samples = len(im_path)-num_val
    # num_samples = 40000
    # train_list=im_path[:num_samples]
    img_rows = 256
    img_cols = 256
    img_channels = 3
    data_loader = Dataloader(dataset_name='shadow',
                                crop_shape=(img_rows, img_cols))

    for i in range(len(path)):
        img_path = path[i]
        img = cv2.imread(img_path)
        img = data_loader.transform_img(img, 256, 256) # Convert to (0,1)
        # Check if transform has error
        img = img[np.newaxis, :]
        start_time = time.clock()
        out_pred = model.predict(img, batch_size=1) # have 1 output images
        end_time = time.clock()
        print('The' + str(i + 1) + 'th image\'s Time:' + str(end_time - start_time))
        
        filename = os.path.basename(img_path).split('.')[0]
        if config_dict['model'] is 'yifan':
            output = out_pred[0].reshape(img_rows, img_cols, img_channels)
            result_img = post_process.rescale_to_image(output)
            save_path = config_dict['save_path'] + "/" + filename + "_enhanced.jpg"
            cv2.imwrite(save_path, result_img)
        else:
            output = out_pred[1].reshape(img_rows, img_cols, img_channels)
            result_img = post_process.rescale_to_image(output)
            save_path = config_dict['save_path'] + "/" + filename + "_enhanced.jpg"
            cv2.imwrite(save_path, result_img)

if __name__ == "__main__":
    test()