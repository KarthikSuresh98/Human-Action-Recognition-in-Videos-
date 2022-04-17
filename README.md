# Human-Action-Recognition-in-Videos

### Abstract
In this project we seek to understand the problem statement of human action recognition in videos. Action recognition has gained traction over the years and with the rise in deep learning, we have been able to get promising results in the field. The aim of the project is to explore different state-of-the-art algorithms in human action recognition in videos, understand the applications and limitations of each and using a chosen model, build a framework that demonstrates the use case of an action
recognition system in a day-to-day life.

The use case we have explored is fall detection wherein the system is expected to raise an alert whenever a fall is detected. The model we have chosen to understand deeply and implement is MoViNets, a family of networks which is computation and memory efficient. The model pretrained on the Kinetics-600 dataset is further trained using the UR fall detection dataset to help the model understand fall and other daily life activities. In the simple system we have implemented, given an input video, if the trained model detects a fall in the video, alert messages are triggered and sent to registered devices to notify the same.


#### Requirements
1. torch 1.10.0 with appropriate cuda distribution
2. Opencv2
3. einops
4. fvcore


#### Dataset
The dataset used for training a fall detection system from MoViNets framework is UR fall detection dataset. Further preprocessing has been done to the datasetfor training the model.

To download the original dataset use : http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html

To download our version: https://drive.google.com/drive/folders/1kRO4sTQ11-q_XewxAuV7XmM15V95_1RT?usp=sharing

#### Training the network
Run the command
~~~
$ python main.py --data_dir '<path to dataset directory>'
~~~

#### Demo
To demo the fall detection system, 
- You need to go to https://www.pushbullet.com/
- Generate a api token as well as register your device so that the alert messages can be send

Now, run the command
~~~
$ python3 demo.py --video_loc 'test_videos/fall_dataset.avi' --api_token '<your api token>'
~~~

Link to a sample demo of the system : https://drive.google.com/file/d/194FPGYxm59BquJBVUoDKX5qo_5E3ddfF/view?usp=sharing

Note : The trained weights (weights/fall_weights.pth) on the UR fall detection dataset is trained using gpu. 

#### References
1. Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, Boqing Gong. MoViNets: Mobile Video Networks for Efficient Video Recognition. Computer Vision and Pattern Recognition (CVPR) , 2021
2. Bogdan Kwolek, Michal Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607
3. https://github.com/Atze00/MoViNet-pytorch
