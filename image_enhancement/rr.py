import shutil
import natsort
import glob
import os
import cv2
import numpy as np

outPath = "segmentation/real_images/"
only_dir="segmentation/JPEGImages/"
TEST_IMAGE_PATHS = glob.glob(os.path.join(only_dir, '*.jpg'))
TEST_IMAGE_PATH=natsort.natsorted(TEST_IMAGE_PATHS, reverse=False)
i=0
while i <len(TEST_IMAGE_PATHS):
    imname=TEST_IMAGE_PATHS[i]
    img=cv2.imread(imname)
    cv2.imwrite(outPath+str(i)+".jpg",img)
    i=i+1
