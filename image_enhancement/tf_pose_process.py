import logging
import sys
import os
import time
import glob 
import natsort
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
from keras.models import load_model

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import test as test
import post_process as post_process
import dense_net as RDN
from data_load import Dataloader

### Get the pose estimation module ###
logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
model = "mobilenet_thin"
resize_out_ratio = 8.0
w = 256
h = 256
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


def avg_paf_dist(clear_vec_x, clear_vec_y, other_vec_x, other_vec_y, w, h):
    clear_paf = np.stack([clear_vec_x, clear_vec_y])
    other_paf = np.stack([other_vec_x, other_vec_y])
    paf_diff = np.linalg.norm(clear_paf - other_paf, axis=0, keepdims=True)
    return np.sum(paf_diff) / (w*h)

def show_detection(enhanced_path, plot_flag = False):
    basename = os.path.basename(enhanced_path).split(".")[0].split('_')[1]
    shadow_path = "/media/tyf/software/ShadowData/medium_ssim/" + enhanced_path.split('/')[-3] + "/" + enhanced_path.split('/')[-2] + "/shadow/shadow_"+basename + ".jpg"
    clear_path = os.path.dirname(os.path.dirname(shadow_path)) + "/clear/clear_" + basename + ".jpg"

    clear_img_ori = common.read_imgfile(clear_path, w, h)
    shadow_img_ori = common.read_imgfile(shadow_path, w, h)
    enhanced_img_ori = common.read_imgfile(enhanced_path, w, h)
    if clear_img_ori is None:
        logger.error('Clear Image can not be read, path=%s' % clear_path)
        sys.exit(-1)
    if shadow_img_ori is None:
        logger.error('Shadow Image can not be read, path=%s' % shadow_path)
        sys.exit(-1)
    if enhanced_img_ori is None:
        logger.error('Enhanced Image can not be read, path=%s' % shadow_path)
        sys.exit(-1)

    ## Clear Image
    t = time.time()
    clear_humans = e.inference(clear_img_ori, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    clear_time = time.time() - t
    logger.info('inference clear image: %s in %.4f seconds.' % (clear_path, clear_time))
    if len(clear_humans) == 0:
        clear_num_keypoints = 0
    else:
        clear_num_keypoints = len(clear_humans[0].body_parts.keys())
    clear_img = TfPoseEstimator.draw_humans(clear_img_ori, clear_humans, imgcopy=True)
    clear_paf = e.pafMat.transpose((2, 0, 1))
    clear_vec_x = np.amax(np.absolute(clear_paf[::2, :, :]), axis=0)
    clear_vec_y = np.amax(np.absolute(clear_paf[1::2, :, :]), axis=0)

    ## Shadow Image
    t = time.time()
    shadow_humans = e.inference(shadow_img_ori, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    shadow_time = time.time() - t
    logger.info('inference Shadow image: %s in %.4f seconds.' % (shadow_path, shadow_time))
    if len(shadow_humans) == 0:
        shadow_num_keypoints = 0
    else:
        shadow_num_keypoints = len(shadow_humans[0].body_parts.keys()) 
    shadow_img = TfPoseEstimator.draw_humans(shadow_img_ori, shadow_humans, imgcopy=True)
    shadow_paf = e.pafMat.transpose((2, 0, 1))
    shadow_vec_x = np.amax(np.absolute(shadow_paf[::2, :, :]), axis=0)
    shadow_vec_y = np.amax(np.absolute(shadow_paf[1::2, :, :]), axis=0)

    ## Enhanced Image
    t = time.time()
    enhanced_humans = e.inference(enhanced_img_ori, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    enhanced_time = time.time() - t
    logger.info('inference Enhanced image in %.4f seconds.' % (enhanced_time))
    if len(enhanced_humans) == 0:
        enhanced_num_keypoints = 0
    else:
        enhanced_num_keypoints = len(enhanced_humans[0].body_parts.keys()) 
    enhanced_img = TfPoseEstimator.draw_humans(enhanced_img_ori, enhanced_humans, imgcopy=True)
    enhanced_paf = e.pafMat.transpose((2, 0, 1))
    enhanced_vec_x = np.amax(np.absolute(enhanced_paf[::2, :, :]), axis=0)
    enhanced_vec_y = np.amax(np.absolute(enhanced_paf[1::2, :, :]), axis=0)

    if plot_flag:
        fig = plt.figure(1)
        plt.imshow(cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB))
        plt.title("Clear Image skeleton")

        fig = plt.figure(2)
        plt.imshow(cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB))
        plt.title("Shadow Image skeleton")

        fig = plt.figure(3)
        plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
        plt.title("Enhanced Image skeleton")

        fig = plt.figure(4)
        plt.imshow(cv2.cvtColor(clear_img_ori, cv2.COLOR_BGR2RGB))
        plt.quiver(clear_vec_x, clear_vec_y, color='b', clim=(0.1, 1.0))
        # plt.quiver(clear_paf[i*2,:,:], clear_paf[i*2+1,:,:], color='b', clim=(0.3, 1.0))
        plt.title("Clear Image PAF")

        fig = plt.figure(5)
        plt.imshow(cv2.cvtColor(shadow_img_ori, cv2.COLOR_BGR2RGB))
        plt.quiver(shadow_vec_x, shadow_vec_y, color='b', clim=(0.1, 1.0))
        # plt.quiver(shadow_paf[i*2,:,:], shadow_paf[i*2+1,:,:], color='b', clim=(0.3, 1.0))
        plt.title("Shadow Image PAF")

        fig = plt.figure(6)
        plt.imshow(cv2.cvtColor(enhanced_img_ori, cv2.COLOR_BGR2RGB))
        plt.quiver(enhanced_vec_x, enhanced_vec_y, color='b', clim=(0.1, 1.0))
        # plt.quiver(enhanced_paf[i*2,:,:], enhanced_paf[i*2+1,:,:], color='b', clim=(0.3, 1.0))
        plt.title("Enhanced Image PAF")

        plt.show()
        plt.close('all')
        # Add function to draw paf

        """
        clear_img = TfPoseEstimator.draw_humans(clear_img, clear_humans, imgcopy=False)
        fig2 = plt.figure(2)
        plt.title("Result")
        plt.imshow(cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB))
        plt.savefig(os.path.dirname(clear_img_name)+"/"+os.path.basename(clear_img_name).split(".")[0]+"_pose.jpg")
        plt.close(fig2)

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)test.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(clear_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        
        plt.imshow(clear_vec_x, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(clear_vec_y, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.savefig(os.path.dirname(clear_img_name)+"/"+os.path.basename(clear_img_name).split(".")[0]+"_tf.jpg")
        plt.show()
        """

    print("Pause")
    return

def get_detection_metric(clear_data_path, other_data_path, enhanced_flag=False):
    assert(len(clear_data_path) == len(other_data_path))
    #Read the data from pickle files and calculate the corresponding metric
    ### Read clear data and shadow/enhanced data ###
    with open("./config/post_process_config.yaml", 'r') as f:
        config_file = f.read()
        config_dict = yaml.load(config_file)  # 用load方法转字典
    metric = config_dict['get_enhanced_images_params']['metric']
    
    ##### Default parameters #####
    w = 256
    h = 256
    num_data_set = 2
    num_data = np.zeros(2)
    ratio = np.zeros(num_data_set+1)
    paf_dist = np.zeros(num_data_set+1)

    # mean average precision considering only detected points
    # low_ssim, medium_ssim, all
    all_tp = np.zeros(num_data_set+1)
    all_fp = np.zeros(num_data_set+1)
    all_fn = np.zeros(num_data_set+1)

    # precision considering the location of detected points
    pcp_thresh = 28.39/2
    tp_points = np.zeros((num_data_set+1, 18))
    fp_points = np.zeros((num_data_set+1, 18))

    cnt = np.zeros(num_data_set+1)
    failed_detect_cnt = np.zeros(num_data_set+1)
    # 循环对于每一片数据进行处理
    for ind in range(len(clear_data_path)):
        with open(clear_data_path[ind], 'rb') as f:
            clear_data = pickle.load(f)
        with open(other_data_path[ind], 'rb') as f:
            other_data = pickle.load(f)

        for img_keys, human in clear_data.items():
            # Find the corresponding other data
            num = img_keys.split('/')[-1].split('.')[0].split('_')[1]
            if img_keys.split("/")[0] == 'medium_ssim':
                ind = 1
            elif img_keys.split("/")[0] == 'low_ssim':
                ind = 0
            
            if img_keys == 'ave_seg_length':
                continue
            # TODO 按照自己的路径进行修改，目标是根据clear_img的key找到对应shadow/enhanced images的key
            if "shadow" in other_data_path[ind].split("/")[-1]:
                other_data_key = os.path.join(img_keys.split('/')[0], img_keys.split('/')[1], img_keys.split('/')[2], 'shadow', 'shadow_' + num +'.jpg')
            elif 'enhanced' in other_data_path[ind].split("/")[-1]:
                other_data_key = os.path.join(img_keys.split('/')[0], img_keys.split('/')[1], img_keys.split('/')[2], 'enhanced_' + num + '.jpg')
            else:
                raise Exception
            if other_data_key.split("/")[1] == 'hands_on_waist':
                tmp = other_data_key.split("/")
                other_data_key = os.path.join(tmp[0], 'hands', tmp[2], tmp[3])
            if other_data_key not in other_data.keys():
                failed_detect_cnt[ind] += 1
                failed_detect_cnt[-1] += 1
                continue
            else:
                other_human = other_data[other_data_key]

            if "paf_dist" in metric:
                tmp = avg_paf_dist(human['vec_x'], human['vec_y'], other_human['vec_x'], other_human['vec_y'], w, h)
                paf_dist[ind] += tmp
                paf_dist[-1] += tmp
            if "detection_rate" in metric:
                # Calculate number of detected keypoints for clear images and shadow/enhanced images
                other_num_keypoints = len(other_human['body_parts'].keys())
                clear_num_keypoints = len(human['body_parts'].keys())
                if clear_num_keypoints != 0:
                    ratio[ind] += other_num_keypoints / clear_num_keypoints
                    ratio[-1] += other_num_keypoints / clear_num_keypoints
            if "point_precision" in metric:
                true_positive = [x for x in other_human['body_parts'].keys() if x in human['body_parts'].keys()]
                false_positive = [x for x in other_human['body_parts'].keys() if x not in human['body_parts'].keys()]
                false_negative = [x for x in human['body_parts'].keys() if x not in other_human['body_parts'].keys()]
                all_tp[ind] += len(true_positive)
                all_tp[-1] += len(true_positive)
                all_fp[ind] += len(false_positive)
                all_fp[-1] += len(false_positive)
                all_fn[ind] += len(false_negative)
                all_fn[-1] += len(false_negative)


            centers = {}
            for i in range(18):
                if i not in human['body_parts'].keys():
                    continue            
                body_part = human['body_parts'][i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                centers[i] = center

            other_centers = {}
            for i in range(18):
                if i not in  other_human['body_parts'].keys():
                    continue
                body_part = other_human['body_parts'][i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                other_centers[i] = center
            
            if "PCPm" in metric:
                for key, clear_center in centers.items():
                    if key not in other_centers.keys():
                        fp_points[ind, int(key)] += 1
                    else:
                        # calculate the distance
                        l2_dist = np.sqrt((clear_center[0] - other_centers[key][0])**2 + (clear_center[1] - other_centers[key][1])**2)
                        if l2_dist <= pcp_thresh:
                            tp_points[ind, int(key)] += 1
                            tp_points[-1, int(key)] += 1
                        else:
                            fp_points[ind, int(key)] += 1
                            fp_points[-1, int(key)] += 1
            
            # 统计个数
            cnt[ind] += 1
            cnt[-1] += 1

    for i in range(num_data_set+1):
        if i == 0:
            print("Dataset low_ssim cnt:{}\n".format(cnt[i]))
        elif i == 1:
            print("Dataset medium_ssim cnt:{}\n".format(cnt[i]))
        else:
            print("All dataset cnt:{}\n".format(cnt[-1]))

        if "detection_rate" in metric:
            ratio[i] /= cnt[i]
            print("Detection Rate:{}".format(ratio[i]))
        if "paf_dist" in metric:
            paf_dist[i] /= cnt[i]
            print("Average PAF distance:{}".format(paf_dist[i]))
        if "point_precision" in metric:
            precision = all_tp[i] / (all_tp[i] + all_fp[i])
            recall = all_tp[i] / (all_tp[i] + all_fn[i])
            print("Point precision:{}, recall:{}".format(precision, recall))
        if "PCPm" in metric:
            all_precision = np.sum(tp_points[i]) / (np.sum(tp_points[i]) + np.sum(fp_points[i]))
            point_precision = tp_points[i] / (tp_points[i] + fp_points[i])
            for i in range(18):
                print('Keypoint {}, precision:{}'.format(i, point_precision[i]))
            print("All precision: {}".format(all_precision))
    print("Cnt is")
    print(cnt)
    print("failed_detect cnt is")
    print(failed_detect_cnt)
    return 0

def save_detection_results(path, save_path, plot_flag = False, ave_seg_flag = False, enhanced_data_flag = False):
    """
    Utilize OpenPose on each images and save the related data
    """
    data = {}
    num_seg = 0
    ave_seg_length = 0
    cnt = 1
    for clear_img_name in path:
        cnt += 1
        print("Process {}".format(cnt / len(path)))
        clear_img = common.read_imgfile(clear_img_name, w, h)
        if clear_img is None:
            logger.error('Clear Image can not be read, path=%s' % clear_img_name)
            sys.exit(-1)
        humans = e.inference(clear_img, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        if (len(humans) == 0):
            continue
        tmp2 = e.pafMat.transpose((2, 0, 1))
        clear_vec_x = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        clear_vec_y = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
        if plot_flag:
            clear_img = TfPoseEstimator.draw_humans(clear_img, humans, imgcopy=False)
            fig2 = plt.figure(2)
            plt.title("Result")
            plt.imshow(cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB))
            plt.savefig(os.path.dirname(clear_img_name)+"/"+os.path.basename(clear_img_name).split(".")[0]+"_pose.jpg")
            plt.close(fig2)

            fig = plt.figure()
            a = fig.add_subplot(2, 2, 1)
            a.set_title('Result')
            plt.imshow(cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB))

            bgimg = cv2.cvtColor(clear_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # show network output
            a = fig.add_subplot(2, 2, 2)
            plt.imshow(bgimg, alpha=0.5)
            tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 2, 3)
            a.set_title('Vectormap-x')
            
            plt.imshow(clear_vec_x, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 2, 4)
            a.set_title('Vectormap-y')
            plt.imshow(clear_vec_y, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()
            plt.savefig(os.path.dirname(clear_img_name)+"/"+os.path.basename(clear_img_name).split(".")[0]+"_tf.jpg")
            plt.show()
            plt.close('all')
        if enhanced_data_flag:
            spl = clear_img_name.split("/")[-1].split("_")
            img_key = spl[0]+'_'+spl[1] + '/' + os.path.join(spl[-4],spl[-3])+ '/' + spl[-2] + '_' + spl[-1]
        else:
            img_key = os.path.join(clear_img_name.split("/")[-5], clear_img_name.split("/")[-4], clear_img_name.split("/")[-3], clear_img_name.split("/")[-2], clear_img_name.split("/")[-1])
        data[img_key] = {'body_parts':humans[0].body_parts, 'vec_x':clear_vec_x, 'vec_y':clear_vec_y}

        # Calculate the mean segment length 
        human = humans[0]  
        if ave_seg_flag: 
            centers = {}
            for i in range(18):
                if i not in human.body_parts.keys():
                    continue            
                body_part = human.body_parts[i]
                center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
                centers[i] = center

            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in humans[0].body_parts.keys():
                    continue
                dx = centers[pair[0]][0] - centers[pair[1]][0]
                dy = centers[pair[0]][1] - centers[pair[1]][1]
                ave_seg_length += np.sqrt(dx**2 + dy**2)
                num_seg += 1

        if cnt % 1000 == 0:
            if num_seg != 0:
                data['ave_seg_length'] = ave_seg_length / num_seg
            else:
                data['ave_seg_length'] = 0
            with open(os.path.dirname(save_path) + "/" + os.path.basename(save_path).split(".")[0] + "_" + str(cnt) + ".pkl", 'wb') as f:
                pickle.dump(data, f)
                print("Save data, cnt: {}".format(cnt))
            data.clear()

    if ave_seg_flag:
        ave_seg_length /= num_seg
        print("Average segment length:{}".format(ave_seg_length))

    data['ave_seg_length'] = ave_seg_length
    with open(os.path.dirname(save_path) + "/" + os.path.basename(save_path).split(".")[0] + "_" + str(cnt) + ".pkl", 'wb') as f:
        pickle.dump(data, f)
        print("Save data successfully!")

def get_enhanced_images(test_config, shadow_paths):
    # Default parameters for images
    img_rows = 256
    img_cols = 256
    img_channels = 3
 
    # Build Model
    model = test.build_test_model(test_config['model'], test_config['model_weight_path'])
    data_loader = Dataloader(dataset_name='shadow',
                            crop_shape=(img_rows, img_cols))

    save_folder = os.path.dirname(test_config['model_weight_path']) + "/enhanced_imgs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(shadow_paths)):
        img_path = shadow_paths[i]
        img = cv2.imread(img_path)
        if img is None: 
            continue

        img = data_loader.transform_img(img, img_rows, img_cols) # Convert to (0,1)
        img = img[np.newaxis, :]
        out_pred = model.predict(img, batch_size=1)
        output = out_pred[0].reshape(img_rows, img_cols, img_channels)
        result_img = post_process.rescale_to_image(output)
        
        spl = img_path.split("/")
        save_path = save_folder + "/" + spl[-5] + "_" + spl[-4] + "_" + spl[-3] + "_enhanced_" + spl[-1].split(".")[0].split("_")[1] + ".jpg"
        cv2.imwrite(save_path, result_img)

def interface1(config_dict):
    # 处理每个数据集的clear_image和shadow_image
    with open(config_dict['data_path'], 'rb') as f:
        data_set = pickle.load(f)
    
    if config_dict['rearrange_path'] == True:
        test_list = data_set['test_list']

        # 修改成本机的目录地址
        clear_paths = []
        shadow_paths = []
        prefix_path = config_dict['prefix_path']
        for path in test_list:
            spl = path.split("/")
            shadow_path = prefix_path + os.path.join(spl[-5], spl[-4], spl[-3], spl[-2], spl[-1])
            clear_path = prefix_path  + os.path.join(spl[-5], spl[-4], spl[-3]) + "/clear/clear_" + spl[-1].split(".")[0].split("_")[1] + ".jpg"
            # if not os.path.exists(shadow_path):
                # raise Exception
            # if not os.path.exists(clear_path):
                # raise Exception
            clear_paths.append(clear_path)
            shadow_paths.append(shadow_path)
    else:
        clear_paths = data_set['clear_paths']
        shadow_paths = data_set['shadow_paths']

    # Detection on clear images and the original shadow images
    save_detection_results(clear_paths, save_path=config_dict['save_path'] + "/clear_pose.pkl", plot_flag=False, ave_seg_flag=True)
    save_detection_results(shadow_paths, save_path=config_dict['save_path'] + "/shadow_pose.pkl", plot_flag=False)

# 输入一个训练的model结果，得到对应的enhanced_img和pose_estimation的结果
def interface2(config_dict):
    # 得到shadow_images的测试集的本机地址
    with open(config_dict['data_path'], 'rb') as f:
        data_set = pickle.load(f)
    if config_dict['rearrange_path'] == True:
        test_list = data_set['test_list']

        # 将服务器上的shadow_data地址换成本机的shadow_data的地址
        shadow_paths = []
        prefix_path = config_dict['prefix_path']
        for path in test_list:
            spl = path.split("/")
            shadow_path = prefix_path + os.path.join(spl[-5], spl[-4], spl[-3], spl[-2], spl[-1])
            if not os.path.exists(shadow_path):
                raise Exception
            shadow_paths.append(shadow_path)
    else:
        shadow_paths = data_set['shadow_paths']

    ### Get enhanced images ###
    get_enhanced_images(config_dict, shadow_paths)
    save_folder = os.path.dirname(config_dict['model_weight_path']) + "/enhanced_imgs"
    if not os.path.exists(save_folder):
        raise Exception
    # 根据shadow_paths 生成enhanced_paths
    enhanced_paths = glob.glob(save_folder+"/*.jpg")
    enhanced_paths = []
    for path in shadow_paths:
        enhance_path = save_folder + "/" + path.split("/")[-5] +'_' + path.split("/")[-4] + '_' + path.split("/")[-3] + '_enhanced_' + path.split("/")[-1].split('.')[0].split('_')[1] + ".jpg"
        enhanced_paths.append(enhance_path)
    
    # 保存pose estimation results
    save_detection_results(enhanced_paths, save_path=os.path.dirname(save_folder)+"/enhanced_pose.pkl", enhanced_data_flag=True)

def post_process():
    with open("./config/post_process_config.yaml", 'r') as f:
        config_file = f.read()
        config_dict = yaml.load(config_file)
    
    if config_dict['function'] == 'save_detection_results':
        interface1(config_dict['save_detection_results_params'])
    elif config_dict['function'] == 'get_enhanced_images':
        interface2(config_dict['get_enhanced_images_params'])
        clear_data_path = config_dict['clear_data_path']
        shadow_data_path = config_dict['shadow_data_path']

        enhanced_data_path = os.path.dirname(config_dict['model_weight_path']) + "/enhanced_imgs"

        clear_data = natsort.natsorted(glob.glob(clear_data_path+"/clear_pose_*.pkl"), reverse=False)
        shadow_data = natsort.natsorted(glob.glob(shadow_data_path+"/shadow_pose_*.pkl"),reverse=False)
        enhanced_data = natsort.natsorted(glob.glob(enhanced_data_path+"/*.pkl"),reverse=False)

        get_detection_metric(clear_data_path=clear_data, other_data_path=enhanced_data, enhanced_flag=True)
    else:
        print("Get the wrong function type")
        print("Available function type is \n {} \n {} \n {} \n {}".format("save_detection_results", "all"))

def failed_detection():
    """
    Find the detection that failed on enhanced images
    """
    hazy_folder = "/home/tyf/Documents/image_enhancement/results/thesis_result/0902_yifan_RDN5_3DB_hazy/enhanced_imgs"
    shadow_folder = "/home/tyf/Documents/image_enhancement/results/thesis_result/all_shadow/ssim_mse_perceptual/enhanced_imgs"
    clear_folder = "/media/tyf/software/ShadowData/"

    hazy_paths = glob.glob(hazy_folder + "/*.jpg")
    for hazy_path in hazy_paths:
        hazy_img = common.read_imgfile(hazy_path, w, h)
        if hazy_img is None:
            logger.error('Clear Image can not be read, path=%s' % hazy_path)
            sys.exit(-1)

        shadow_img = common.read_imgfile(shadow_folder+"/"+os.path.basename(hazy_path), w, h)
        if shadow_img is None:
            logger.error('Clear Image can not be read, path=%s' % shadow_folder+"/"+os.path.basename(hazy_path))
            sys.exit(-1)
        
        tmp = hazy_path.split("/")[-1].split('.')[0].split('_')
        if len(tmp) == 6:
            clear_path = clear_folder + tmp[0]+'_'+tmp[1] + '/'+tmp[2]+'/'+tmp[3]+'/clear/clear_' + tmp[-1]+'.jpg'
        elif len(tmp) == 7:
            clear_path = clear_folder + tmp[0]+'_'+tmp[1] + '/'+tmp[2]+'_'+tmp[3]+'/'+tmp[4]+'/clear/clear_' + tmp[-1]+'.jpg'
        elif len(tmp) == 8:
            clear_path = clear_folder + tmp[0]+'_'+tmp[1] + '/'+tmp[2]+'_'+tmp[3]+'_'+tmp[4]+'/'+tmp[5]+'/clear/clear_' + tmp[-1]+'.jpg'


        if not os.path.exists(clear_path):
            print("Can't find clear image path:{}".format(clear_path))
        clear_img = common.read_imgfile(clear_path, w, h)
        if clear_img is None:
            logger.error('Clear Image can not be read, path=%s' % clear_path)
            sys.exit(-1)


        hazy_humans = e.inference(hazy_img, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        shadow_humans = e.inference(shadow_img, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        clear_humans = e.inference(clear_img, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        
        if (len(hazy_humans) == 0 or len(shadow_humans) == 0 or len(clear_humans) == 0):
            continue
        hazy_body = hazy_humans[0].body_parts
        shadow_body = shadow_humans[0].body_parts
        clear_body = clear_humans[0].body_parts
        if len(shadow_body.keys()) < len(clear_body.keys()):
            print("Find failed cases. ")
            print(shadow_folder+"/"+os.path.basename(hazy_path))

        # if 0 in shadow_body.keys() and 0 not in hazy_body.keys():
        #     print("Find path for 0")
        #     print(hazy_path)
        #     print(shadow_folder+"/"+os.path.basename(hazy_path))
        
        # if 5 in shadow_body.keys() and 5 not in hazy_body.keys():
        #     print("Find path for 5")
        #     print(hazy_path)
            # print(shadow_folder+"/"+os.path.basename(hazy_path))

def show_single_detection(path, plot_flag=False):
    img = common.read_imgfile(path, w, h)
    if img is None:
        logger.error('Clear Image can not be read, path=%s' % path)
        sys.exit(-1)
    humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    if len(humans) == 0:
        num_keypoint = 0
    else:
        num_keypoint = len(humans[0].body_parts.keys())

    paf = e.pafMat.transpose((2, 0, 1))
    vec_x = np.amax(np.absolute(paf[::2, :, :]), axis=0)
    vec_y = np.amax(np.absolute(paf[1::2, :, :]), axis=0)

    if plot_flag:
        fig2 = plt.figure(2)
        plt.title("Result")
        img_copy = TfPoseEstimator.draw_humans(img, humans, imgcopy=True)
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.savefig(os.getcwd()+"/pose.jpg")
        plt.close(fig2)

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title("Result")
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        
        plt.imshow(vec_x, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(vec_y, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.savefig(os.getcwd()+"/result.jpg")
        plt.close('all')

if __name__ == "__main__":
    post_process()