from model import *
import torch.nn as nn
from config import _C

import torch
import torch.nn.functional as F

import argparse
import numpy as np
from dataloader import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'data/')
args = parser.parse_args()

def main():
  
    num_epochs = 5

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  

    model = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True)
    model.classifier[3] = ConvBlock3D(2048,
                            2,
                            kernel_size=(1, 1, 1),
                            tf_like=True,
                            causal=True,
                            conv_type="3d",
                            bias=True)

    dataset_train = Dataset(args.data_dir, train = True)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = 1, shuffle=True, num_workers=0)

    dataset_test = Dataset(args.data_dir, train = False)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=0.1)

    total_train_images = dataset_train.__len__()
    total_test_images = dataset_test.__len__()

    model.to(device=device)
    for epoch in range(num_epochs):

      model.train()
      avg_train_loss = 0
      train_count = 0

      num_frames_per_clip = 8
      model.clean_activation_buffers()
      optimizer.zero_grad()

      for i, data in enumerate(train_dataloader):
          all_frames,target = data
          target = target[0]

          input_video = rearrange(torch.from_numpy(all_frames.numpy()), "b t h w c-> b c t h w")

          num_clips = int(input_video.shape[2]/num_frames_per_clip)

          input_video = input_video.to(device=device)
          target = target.to(device=device)
          
          for j in range(num_clips):
            frame = input_video[:, :, j*num_frames_per_clip:(j+1)*num_frames_per_clip]
            
            out = F.softmax(model(frame) , dim = 1)
            loss = F.binary_cross_entropy(out, target)/num_clips
                        
            loss.backward()

            avg_train_loss = avg_train_loss + loss.data

          optimizer.step()
          optimizer.zero_grad()
      
          #clean the buffer of activations
          model.clean_activation_buffers()  
          train_count = train_count + 1
          print('Processed %d / %d training videos' %(train_count, total_train_images))

      print('Epoch %d - Training loss : %f' %(epoch, (avg_train_loss/train_count)))  

      print('\n----------------------------------------------\n')
      print('saving net...')
      torch.save(model.state_dict(), 'weights/network_' + str(epoch) + '.pth')    
      
      model.eval()
      model.clean_activation_buffers()
      avg_test_loss = 0
      test_count = 0

      for i, data in enumerate(test_dataloader):
          all_frames,target = data
          target = target[0]
                    
          input_video = rearrange(torch.from_numpy(all_frames.numpy()), "b t h w c-> b c t h w")

          input_video = input_video.to(device=device)
          target = target.to(device=device)


          for j in range(input_video.shape[2]):
              
              out = model(input_video[:,:,[j]])
              loss = F.binary_cross_entropy(F.softmax(out, dim = 1), target)/input_video.shape[2]

              avg_test_loss = avg_test_loss + loss.data
          
          prediction = F.softmax(out, dim = 1)
          print(torch.argmax(prediction))

          #clean the buffer of activations
          model.clean_activation_buffers()  
          test_count = test_count + 1
          print('Processed %d / %d test videos' %(test_count, total_test_images))     

      print('Test loss : ' , (avg_test_loss/test_count))
      print('\n-----------------------------------------------\n') 


if __name__ == '__main__':
  main()
