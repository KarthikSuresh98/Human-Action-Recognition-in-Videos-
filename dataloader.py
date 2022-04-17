import torch
import random
import os
from utils import DataAugment
import numpy as np
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train = True, image_size = 172):

      self.train = train
      self.image_size = image_size

      random.seed(1)

      labels = os.listdir(data_dir)[:2]
      all_files = []
      for label in labels:
        loc = os.path.join(data_dir , label)
        label_files = os.listdir(loc)
        label_files = [os.path.join(loc , i) for i in label_files]
        label_files = random.sample(label_files, len(label_files))
        all_files.append(label_files)

      images = []
      num_labels = len(labels)
      for i in range(num_labels):
        num_files = len(all_files[i])
        if self.train:
            images = images + all_files[i][0:int(num_files * 0.8)]
        else:
            images = images + all_files[i][int(num_files * 0.8):]
        
        self.img_names = images

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

      img_name = self.img_names[index]
      cap = cv2.VideoCapture(img_name)
      augmentation = random.randint(1,8)
  
      if self.train == False:
        augmentation = 8

      aug = DataAugment()

      # Check if camera opened successfully
      if (cap.isOpened()== False): 
        print("Error opening video stream or file")

      # Read until video is completed
      all_frames = []
      while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:    
          if augmentation == 1:
            frame = aug.noisy('gauss', frame)  
          elif augmentation == 2:
            frame = aug.noisy('s&p', frame)
          elif augmentation == 3:
            frame = aug.noisy('poisson', frame)
          elif augmentation == 4:
            frame = aug.noisy('speckle', frame)
          elif augmentation == 5:
            frame = aug.brightness(frame, 0.5, 30)
          elif augmentation == 6:
            frame = aug.channel_shift(frame, 60)
          elif augmentation == 7:
            frame = aug.horizontal_flip(frame)
          else:
            pass 
 
          frame = cv2.resize(frame, (self.image_size, self.image_size))
          all_frames.append(frame)
       
        # Break the loop
        else: 
          break

      # When everything done, release the video capture object
      cap.release()
      all_frames = torch.tensor(np.array(all_frames), dtype=torch.float32)/255.0


      label = img_name.split('/')[-2]
      classes = {'adl' : 0, 'fall' : 1}

      target = torch.zeros((1,2) , dtype = torch.float32)
      target[0, classes[label]] = 1.0

      return all_frames , target