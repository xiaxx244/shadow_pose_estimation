import argparse
import sys
import time
import os
import glob
from tf_pose import common
import numpy as np
import cv2
import math
import natsort
#from tf_pose.estimator import TfPoseEstimator
from tf_pose.new_estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

OUTPUT_DIR = 'r_far_female_filter3/'
#count=0

def sort_annotate_and_write(image, human_info, human_info_orig, sorted_idx,sorted_idx_orig,im_file):
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
    cc=0
    bc=0
    l1=[]
    if len(sorted_idx_orig)==0:
        return (-1,-1)
    if len(sorted_idx)==0:
        return (0,0)
    #print(len(sorted_idx_or))
    for idx in sorted_idx:
        human = human_info[idx]
        for (information,location) in human:
            if location[0]!=-1:
                #c=c+1
                l1.append((information,location[0],location[1]))
            else:
                l1.append((information,-1,-1))
            f.write(information+ ": " + str(location[0]) +", "+ str(location[1]) )
            f.write("\n")
        f.write("\n")
    f.close()
    l2=[]

    for idx_orig in sorted_idx_orig:
        human_orig = human_info_orig[idx_orig]
        for (information1,location1) in human_orig:
            if location1[0]!=-1:
                #c=c+1
                l2.append((information1,location1[0],location1[1]))
            else:
                l2.append((information1,-1,-1))
            #f.write(information+ ": " + str(location[0]) +", "+ str(location[1]) )
            #f.write("\n")
        #f.write("\n")
    #f.close()
    l1.sort()
    l2.sort()
    print(l1)
    print(l2)
    print(len(l1))
    for i in range(18):
        a,b,c=l1[i]
        a1,b1,c1=l2[i]
        if abs(b-(256/640)*b1)<=5:
            bc=bc+1
        if b!=-1 or b1==-1:
            cc=cc+1
    #print(c)
    cv2.imwrite(op_imfile.replace(op_imfile[-4:], "op.JPG"), image)
    return (cc,bc)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='image_enhancement/result4')
    parser.add_argument('--image_type', type=str, default='*.png')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()
    input_folder_orig= "../../../../media/bizon/Elements/shadow_test/private/r_far_female_filter3"
    path_orig = glob.glob(input_folder_orig+'/*.*')
    TEST_IMAGE_PATHS_orig = path_orig
    TEST_IMAGE_PATHS_orig=natsort.natsorted(TEST_IMAGE_PATHS_orig,reverse=False)
    input_folder = "../image_enhancement/r_far_female_filter3"
    path = glob.glob(input_folder+'/*.*')
    TEST_IMAGE_PATHS = path
    TEST_IMAGE_PATHS=natsort.natsorted(TEST_IMAGE_PATHS,reverse=False)
    #TEST_IMAGE_PATHS.sort(key=lambda f: int(filter(str.isdigit, f)))

    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    t = time.time()
    count=0
    b_t=0
    skip=0
    length=min(len(TEST_IMAGE_PATHS),len(TEST_IMAGE_PATHS_orig))
    for i in range(length):
        print('Processing: {0}'.format(TEST_IMAGE_PATHS_orig[i]))
        img_orig = common.read_imgfile(TEST_IMAGE_PATHS_orig[i], 256, 256)
        #img_orig=img_orig.resize(256,256)
        img = common.read_imgfile(TEST_IMAGE_PATHS[i], None, None)
        if img is None: continue

        humans_orig = e.inference(img_orig, resize_to_default=False, upsample_size=4)
        humans = e.inference(img, resize_to_default=False, upsample_size=4)
        image_orig, human_info_orig, face_info_orig = TfPoseEstimator.draw_humans(TEST_IMAGE_PATHS_orig[i],img_orig , humans_orig, imgcopy=False)
        image, human_info, face_info = TfPoseEstimator.draw_humans(TEST_IMAGE_PATHS[i], img, humans, imgcopy=False)

        face_x_orig = []
        for face in face_info_orig:
            fx = face[0] if face[0]>0 else 1e4
            face_x_orig.append(fx)

        face_x = []
        for face1 in face_info:
            fx1 = face1[0] if face1[0]>0 else 1e4
            face_x.append(fx1)

        sorted_idx_orig = np.argsort(face_x_orig)
        sorted_idx = np.argsort(face_x)
        (count_temp,b_temp)=sort_annotate_and_write(image, human_info, human_info_orig, sorted_idx, sorted_idx_orig, TEST_IMAGE_PATHS[i])
        if count_temp==-1:
            skip=skip+1
            continue
        count=count+count_temp
        print(count_temp)
        b_t=b_t+b_temp
    print("Processing time for {0} images: {1} sec".format(len(TEST_IMAGE_PATHS), time.time() - t))
    print(count/(18*(len(TEST_IMAGE_PATHS)-skip)))
    print(b_t/count)
