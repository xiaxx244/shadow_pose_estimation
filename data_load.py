import numpy as np
import keras
import random
import scipy
import os
import cv2 as cv
from datetime import datetime

class Dataloader():
    def __init__(self, dataset_name, crop_shape=(256, 256)):
        self.dataset_name = dataset_name
        self.crop_shape = crop_shape
    def imread_color(self, path):
        # Return image array in [r,g,b] format
        img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)/255.
        b, g, r = cv.split(img)
        img_rgb = cv.merge([r, g, b])
        return img_rgb

    def imwrite(self, path, img):
        r, g, b = cv.split(img)
        img_rgb = cv.merge([b, g, r])
        cv.imwrite(path, img_rgb)

    def transform_img(self, img, height, width):
        # transform image in size, and to float number (0, 1)
        im = np.zeros((height, width, 3), dtype='uint8')
        im[:, :, :] = 128
        if img.shape[0] >= img.shape[1]:
            scale = img.shape[0] / height
            new_width = int(img.shape[1] / scale)
            diff = (width - new_width) // 2
            img = cv.resize(img, (new_width, height))
            im[:, diff:diff + new_width, :] = img
        else:
            scale = img.shape[1] / width
            new_height = int(img.shape[0] / scale)
            diff = (height - new_height) // 2
            img = cv.resize(img, (width, new_height))
            im[diff:diff + new_height, :, :] = img
        im = np.float32(im) / 127.5 - 1
        return im

    def load_data(self, fake_list,  batch_size=16):
        n_batches = int(len(fake_list) / batch_size)
        while True:
            random.shuffle(fake_list)
            for i in range(n_batches - 1):
                batch_path = fake_list[i * batch_size:(i + 1) * batch_size]
                input_imgs = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 3), dtype="float32")
                gt = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 3), dtype="float32")

                number = 0
                for img_A_path in batch_path:
                    parent_path = os.path.dirname(os.path.dirname(img_A_path))
                    img_B = cv.imread(parent_path+"/clear/clear_"+os.path.basename(img_A_path).split(".")[0].split("_")[1]+".jpg")
                    if img_B is None:
                        print("imgA path:{}".format(img_A_path))                    
                    
                    img_A = cv.imread(img_A_path)
                    crop_img_A = self.transform_img(img_A,256,256)
                    crop_img_B = self.transform_img(img_B,256,256)

                    input_imgs[number, :, :, :] = crop_img_A
                    gt[number, :, :, :] = crop_img_B
                    number += 1
                yield input_imgs, [gt, gt, gt]

    def load_single_output_data(self, fake_list,  batch_size=16):
        n_batches = int(len(fake_list) / batch_size)
        while True:
            random.shuffle(fake_list)
            for i in range(n_batches - 1):
                batch_path = fake_list[i * batch_size:(i + 1) * batch_size]
                input_imgs = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 3), dtype="float32")
                gt = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 3), dtype="float32")

                number = 0
                for img_A_path in batch_path:
                    parent_path = os.path.dirname(os.path.dirname(img_A_path))
                    img_B = cv.imread(parent_path+"/clear/clear_"+os.path.basename(img_A_path).split(".")[0].split("_")[1]+".jpg")
                    if img_B is None:
                        print("imgA path:{}".format(img_A_path))                    
                    
                    img_A = cv.imread(img_A_path)
                    crop_img_A = self.transform_img(img_A,256,256)
                    crop_img_B = self.transform_img(img_B,256,256)

                    input_imgs[number, :, :, :] = crop_img_A
                    gt[number, :, :, :] = crop_img_B
                    number += 1
                end = datetime.now()
                print("Loading time:{}".format((end-start).seconds))
                yield input_imgs, gt


class DataLoader2(keras.utils.Sequence):
    def __init__(self, data_set, batch_size, dim=(256, 256,3), shuffle=True):
        self.dim = dim
        self.list_IDS = data_set
        self.crop_shape = [dim[0], dim[1]]
        self.batch_size = batch_size
        self.shuffle = shuffle
        # find the reference data
        self.labels = self.generate_labels()
        self.indexes = np.arange(len(self.list_IDS))
        self.on_epoch_end()

    def generate_labels(self):
        # find the position for each shadow image
        labels = {}
        for shadow_img in self.list_IDS:
            parent_path = os.path.dirname(os.path.dirname(shadow_img))
            #img_B = parent_path+"/clear/clear_"+os.path.basename(shadow_img).split(".")[0].split("_")[1]+".jpg"
            img_B = parent_path + "/clear/" + os.path.basename(shadow_img).split("_")[0]+".jpg"
            labels[shadow_img] = img_B
        return labels

    def __len__(self):
        return int(np.floor(len(self.list_IDS) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDS[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim), dtype="float32")
        y = np.empty((self.batch_size, *self.dim), dtype="float32")
        for i, ID in enumerate(list_IDs_temp):
            X[i, :, :, :] = self.transform_img(cv.imread(ID), 256, 256)
            y[i, :, :, :] = self.transform_img(cv.imread(self.labels[ID]), 256, 256)
        
        return X, y

    def transform_img(self, img, height, width):
        # transform image in size, and to float number (0, 1)
        im = np.zeros((height, width, 3), dtype='uint8')
        im[:, :, :] = 128
        if img.shape[0] >= img.shape[1]:
            scale = img.shape[0] / height
            new_width = int(img.shape[1] / scale)
            diff = (width - new_width) // 2
            img = cv.resize(img, (new_width, height))
            im[:, diff:diff + new_width, :] = img
        else:
            scale = img.shape[1] / width
            new_height = int(img.shape[0] / scale)
            diff = (height - new_height) // 2
            img = cv.resize(img, (width, new_height))
            im[diff:diff + new_height, :, :] = img
        im = np.float32(im) / 127.5 - 1
        return im
