import numpy as np
import random
import cv2


class DataAugment():
  def __init__(self):
    pass

  def noisy(self,noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 5
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[tuple(coords)] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[tuple(coords)] = 0
      return out
    
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

  def brightness(self,img, low, high):
      value = random.uniform(low, high)
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      hsv = np.array(hsv, dtype = np.float64)
      hsv[:,:,1] = hsv[:,:,1]*value
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2] = hsv[:,:,2]*value 
      hsv[:,:,2][hsv[:,:,2]>255]  = 255
      hsv = np.array(hsv, dtype = np.uint8)
      img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      return img

  def channel_shift(self,img, value):
      value = int(random.uniform(-value, value))
      img = img + value
      img[:,:,:][img[:,:,:]>255]  = 255
      img[:,:,:][img[:,:,:]<0]  = 0
      img = img.astype(np.uint8)
      return img

  def horizontal_flip(self,img):
      return cv2.flip(img, 1)