from glob import glob
import numpy as np
import random
import scipy
import os
import cv2 as cv

class Dataloader():
    def __init__(self, dataset_name, crop_shape=(256, 256)):
        self.dataset_name = dataset_name
        self.crop_shape = crop_shape
    def imread_color(self, path):
        img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)/255.
        b, g, r = cv.split(img)
        img_rgb = cv.merge([r, g, b])
        return img_rgb

    def imwrite(self, path, img):
        r, g, b = cv.split(img)
        img_rgb = cv.merge([b, g, r])
        cv.imwrite(path, img_rgb)

    def transform_img(self, img, height, width):
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
                    #print(img_A_path.split("_")[0])

                    #print("../segmentation/JPEGImages/"+img_A_path.split("_")[0].split("/")[-1]+".jpg")
                    img_B = cv.imread("../../../../media/bizon/Elements/ITS/train/ITSclear/"+img_A_path.split("_")[0].split("/")[-1]+".png")
                    img_A = cv.imread(img_A_path)




                    crop_img_A = self.transform_img(img_A,256,256)
                    #print(crop_img_A.shape)
                    crop_img_B = self.transform_img(img_B,256,256)

                    input_imgs[number, :, :, :] = crop_img_A
                    gt[number, :, :, :] = crop_img_B
                    number += 1
                yield input_imgs, [gt,gt,gt]
