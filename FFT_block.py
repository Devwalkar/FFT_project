import torch.nn as nn 
import torch 
import numpy as np 

class FFT_block(nn.Module):

    def __init__(self,center_height=0.2,center_width=0.2,
                Real_filter_size=(3,3),Imag_filter_size=(3,3),kernel_depth=50,in_channels=100):

        super(FFT_block,self).__init__()

        self.center_height = center_height
        self.center_width = center_width
        self.real_kernel = nn.Conv2d(in_channels,kernel_depth,Real_filter_size)
        self.img_kernel = nn.Conv2d(in_channels,kernel_depth,Imag_filter_size)

    def forward(self,input_batch):

        input_batch = torch.cat((input_batch.unsqueeze(-1),torch.zeros(input_batch.shape).unsqueeze(-1)),dim=-1)
        input_batch = torch.fft(input_batch,2)

        Real_input,Img_input = self.zeroing_out_low(input_batch)

        Real_input = self.real_kernel(Real_input)
        Img_input = self.img_kernel(Img_input)

        input_batch = torch.cat((Real_input.unsqueeze(-1),Img_input.unsqueeze(-1)),dim=-1)
        input_batch = torch.ifft(input_batch,2)[:,:,:,:,0]

        return input_batch

    def zeroing_out_low(self,input_batch):

        Image_height,Image_width = input_batch.shape[2],input_batch.shape[3]
        Real_input = input_batch[:,:,:,:,0]
        Img_input = input_batch[:,:,:,:,1]

        cut_height_end = int(self.center_height*Image_height)
        cut_width_end = int(self.center_width*Image_width)

        Real_input[:,:,:cut_height_end,:cut_width_end] = torch.zeros((input_batch.shape[0],input_batch.shape[1],int(self.center_height*Image_height),int(self.center_width*Image_width)))

        Img_input[:,:,:cut_height_end,:cut_width_end] = torch.zeros((input_batch.shape[0],input_batch.shape[1],int(self.center_height*Image_height),int(self.center_width*Image_width)))

        return Real_input,Img_input