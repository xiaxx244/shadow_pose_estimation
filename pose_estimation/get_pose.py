import argparse
import sys
import time
import os
import glob
from tf_pose import common
import numpy as np
import cv2

#from tf_pose.estimator import TfPoseEstimator
from tf_pose.new_estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

OUTPUT_DIR = 'original/'

def sort_annotate_and_write(image, human_info, sorted_idx, im_file):
    """ Annotate key-points in image and write to a file
         > also writes person based on sorted index
            (left-most person in the image to the right-most)
    """
    global OUTPUT_DIR
    fname = im_file.split('/')[-1]
    op_imfile = os.path.join(OUTPUT_DIR, fname.replace(fname[-4:], ".txt"))
    f = open(op_imfile, "w")
    f.write(str(len(human_info)))
    f.write("\n\n")

    for idx in sorted_idx:
        human = human_info[idx]
        for (information,location) in human:
            f.write(information+ ": " + str(location[0]) +", "+ str(location[1]) )
            f.write("\n")
        f.write("\n")
    f.close()

    cv2.imwrite(op_imfile.replace(op_imfile[-4:], "op.JPG"), image)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../image_enhancement/IMG_0186/')
    parser.add_argument('--image_type', type=str, default='*.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()

    TEST_IMAGE_PATHS = glob.glob(os.path.join(args.image_dir, args.image_type))
    #TEST_IMAGE_PATHS.sort(key=lambda f: int(filter(str.isdigit, f)))

    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    t = time.time()

    for im_file in TEST_IMAGE_PATHS:
        print('Processing: {0}'.format(im_file))
        img = common.read_imgfile(im_file, None, None)
        if img is None: continue

        humans = e.inference(img, resize_to_default=False, upsample_size=4)
        image, human_info, face_info = TfPoseEstimator.draw_humans(im_file, img, humans, imgcopy=False)

        face_x = []
        for face in face_info:
            fx = face[0] if face[0]>0 else 1e4
            face_x.append(fx)
        sorted_idx = np.argsort(face_x)
        sort_annotate_and_write(image, human_info, sorted_idx, im_file)

    print("Processing time for {0} images: {1} sec".format(len(TEST_IMAGE_PATHS), time.time() - t))
