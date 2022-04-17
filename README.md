# Human-Action-Recognition-in-Videos

### Abstract
In this project we seek to understand the problem statement of human action recognition in videos. Action recognition has gained traction over the years and with the rise in deep learning, we have been able to get promising results in the field. The aim of the project is to explore different state-of-the-art algorithms in human action recognition in videos, understand the applications and limitations of each and using a chosen model, build a framework that demonstrates the use case of an action
recognition system in a day-to-day life.

The use case we have explored is fall detection wherein the system is expected to raise an alert whenever a fall is detected. The model we have chosen to understand deeply and implement is MoViNets, a family of networks which is computation and memory efficient. The model pretrained on the Kinetics-600 dataset is further trained using the UR fall detection dataset to help the model understand fall and other daily life activities. In the simple system we have implemented, given an input video, if the trained model detects a fall in the video, alert messages are triggered and sent to registered devices to notify the same.
