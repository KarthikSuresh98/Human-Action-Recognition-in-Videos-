from model import *
import torch.nn as nn
from config import _C

import torch
import torch.nn.functional as F

import os
import argparse
import cv2
import random
import numpy as np
import requests
import json


parser = argparse.ArgumentParser()
parser.add_argument('--video_loc', default = '')
parser.add_argument('--api_token', default = '')
parser.add_argument('--checkpoint', default= '')
args = parser.parse_args()

def pushbullet_message(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = args.api_token
    resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error',resp.status_code)
    else:
        print ('Message sent') 

def main():

    if args.video_loc == '':
      print('no video file provided')
    
    else:

      model = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True)
      model.classifier[3] = ConvBlock3D(2048,
                              2,
                              kernel_size=(1, 1, 1),
                              tf_like=True,
                              causal=True,
                              conv_type="3d",
                              bias=True)
    
      cap = cv2.VideoCapture(args.video_loc)
      # Check if camera opened successfully
      if (cap.isOpened()== False): 
        print("Error opening video stream or file")

      # Read until video is completed
      all_frames = []
      while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:    
          frame = cv2.resize(frame, (172,172))
          all_frames.append(frame)
        else:
          break    

      all_frames = np.array(all_frames)
      all_frames = torch.tensor(all_frames, dtype = torch.float32)
      all_frames = all_frames.unsqueeze(0)
      all_frames = rearrange(torch.from_numpy(all_frames.numpy()), "b t h w c-> b c t h w")


      all_frames = all_frames.cuda()

      model.cuda()
      model.load_state_dict(torch.load('weights/fall_weights.pth'))

      model.eval()
      model.clean_activation_buffers()

      for j in range(all_frames.shape[2]):
          out = F.softmax(model(all_frames[:,:,[j]]) , dim = 1)
      
      if(torch.argmax(out) == 1):
        pushbullet_message('Emergency!!!', 'Person of Interest had probably had a fall')
        pushbullet_message('Emergency!!!', 'Person of Interest had probably had a fall')
        pushbullet_message('Emergency!!!', 'Person of Interest had probably had a fall')
      else:
        print('Daily Life Activity')
      print('\n')
      
      #clean the buffer of activations
      model.clean_activation_buffers()  

if __name__ == '__main__':
  main()




