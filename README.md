# shadow_pose_estimation
pose estimation with shadow image enhancement
### Usage
- download the **[OTS dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)** and put the dataset in your desired path.
- set up the image path in train.py (specifically the [im_path](https://github.com/xiaxx244/shadow_pose_estimation/blob/4f85ab103af261955c54a4828596a00089156778/image_enhancement/train.py#L194), [img_B path](https://github.com/xiaxx244/shadow_pose_estimation/blob/7771c474cc3ce62ca5067c07dcfb73fb46509d8a/image_enhancement/train.py#L149))
- setup the [image path](https://github.com/xiaxx244/shadow_pose_estimation/blob/56f4205ca331b93165d7dc9a8b1c06db12db0547/image_enhancement/data_load.py#L55) in data_load.py
- setup the [save path](https://github.com/xiaxx244/shadow_pose_estimation/blob/f102cff8163618cdd6a8892ee4c2adbacc00242f/image_enhancement/train.py#L139) for model
- train image enhancement model model by running command 
```
cd image_enhancement
python3 train.py
```
- put the test image path in your desired path
- modifiy the path for [test images](https://github.com/xiaxx244/shadow_pose_estimation/blob/42167948f84482356bcfd1220e277f5780bb203a/image_enhancement/test.py#L128) and [saved model](https://github.com/xiaxx244/shadow_pose_estimation/blob/1a9cb1ed40397dce35a4b448411cb01638fe3470/image_enhancement/test.py#L78) in test.py
- run test.py py to run the inference results for image enhancement
- if you need to run pose estimation results on enhanced images, please run the command
```
cd pose_estimation
python3 get_pose.py
```
### Image Enhancement Network Design
#### Overall arch
<img src="pipeline.png" width=30% height=30%> 

#### EM arch
<img src="aa.jpg" width=50% height=50%> 

### System setup
<img src="res.png" width=80% height=80%> 

### Acknowledgements
- https://github.com/CMU-Perceptual-Computing-Lab/openpose
- https://github.com/xahidbuffon/tf-pose-estimation
- https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2

