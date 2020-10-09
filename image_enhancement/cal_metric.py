"""
This file will calculate different metrics for different dataset
"""
import sys
import os

import cv2
import numpy as np
from libsvm import svmutil
import brisque
import glob


class metric_calculator():
    def __init__(self):
        self.brisque = brisque.BRISQUE()

    def brisque_metric(self, image_path):
        """
        Calculate the brisque metric for an image
        """
        quality_score = self.brisque.get_score(image_path)
        return quality_score

if __name__ == "__main__":
    calculator = metric_calculator()
    base_path = "/media/tyf/software/ShadowData/low_ssim"
    shadow_path = glob.glob(base_path + "/*/*/shadow/*.jpg")
    print("Data size is :{}".format(len(shadow_path)))
    clear_path = []
    for shadow in shadow_path:
        parent_path = os.path.dirname(os.path.dirname(shadow))
        clear = parent_path + "/clear/clear_" + os.path.basename(shadow).split(".")[0].split("_")[1] + ".jpg"
        clear_path.append(clear)
    
    clear_metric = []
    shadow_metric = []
    failed_cnt = 0
    for i in range(len(shadow_path)):
        shadow_score = calculator.brisque_metric(shadow_path[i])
        shadow_metric.append(shadow_score)
        clear_score = calculator.brisque_metric(clear_path[i])
        clear_metric.append(clear_score)
        # 统计 clear_metric 大于 shadow_metric的个数
        if (clear_score > shadow_score):
            failed_cnt += 1
        print('\r'+str(i/len(shadow_path))+'%', end='')

    clear_metric = np.asarray(clear_metric)
    shadow_metric = np.asarray(shadow_metric)
    ave_clear_metric = np.mean(clear_metric)
    ave_shadow_metric = np.mean(shadow_metric)
    print("Clear average metric:{}".format(ave_clear_metric))
    print("Shadow average metric:{}".format(ave_shadow_metric))
    print("Number of failed ratio:{}".format(failed_cnt / len(shadow_path)))