import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default='')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()

    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    t = time.time()

    vid_src = 0 if args.camera=='' else args.camera
    cam = cv2.VideoCapture(vid_src)
    if not cam.isOpened(): print("Error opening camera or video file")
    else: fno = 0

    while True:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=False, upsample_size=4)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break
        fno += 1

    print("Processing time for {0} images: {1} sec".format(fno, time.time()-t))
    cv2.destroyAllWindows()
