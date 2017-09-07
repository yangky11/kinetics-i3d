import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math


def get_padding(in_duration, in_height, in_width, kernel_size, stride):
    print((in_duration, in_height, in_width, kernel_size, stride))
    out_duration = math.ceil(in_duration / float(stride[0]))
    out_height = math.ceil(in_height / float(stride[1]))
    out_width  = math.ceil(in_width / float(stride[2]))

    if in_duration % stride[0] == 0:
      pad_duration = max(kernel_size[0] - stride[0], 0)
    else:
      pad_duration = max(kernel_size[0] - (in_duration % stride[0]), 0)
    if in_height % stride[1] == 0:
      pad_height = max(kernel_size[1] - stride[1], 0)
    else:
      pad_height = max(kernel_size[1] - (in_height % stride[1]), 0)
    if in_width % stride[2] == 0:
      pad_width = max(kernel_size[2] - stride[2], 0)
    else:
      pad_width = max(kernel_size[2] - (in_width % stride[2]), 0)

    pad_before = pad_duration // 2
    pad_after = pad_duration - pad_before
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_before
    print((pad_duration, pad_height, pad_width))
    assert pad_before == pad_after
    assert pad_top == pad_bottom
    assert pad_left == pad_left
   
    return(pad_before, pad_top, pad_left)



class Unit3D(nn.Module):
  
  def __init__(self, in_shape,
               out_channels, 
               kernel_size=(1, 1, 1), 
               stride=(1, 1, 1), 
               activation_fn=F.relu, 
               batch_norm=True, 
               bias=False):
    super().__init__()
    self.in_shape = in_shape
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.activation_fn = activation_fn
    self.batch_norm = batch_norm
    self.bias = bias

    assert len(in_shape) == 5 
    in_duration = in_shape[2]
    in_height = in_shape[3]
    in_width = in_shape[4]
    
    padding = get_padding(in_duration, in_height, in_width, kernel_size, stride)
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                          stride, padding, bias=self.bias)
    if self.batch_norm:
      self.batchnorm = nn.BatchNorm3d(in_shape[1])

 
  def forward(self, x):
    x = self.conv(x)
    if self.batch_norm:
      x = self.batchnorm(x)
    if self.activation_fn != None:
      x = self.activation_fn(x)
    return x



class InceptionI3d(nn.Module):

  def __init__(self, in_shape, num_classes=400):
    super().__init__()
    assert len(in_shape) == 5
    self.in_shape = in_shape
    self.num_classes = num_classes

    self.Conv3d_1a_7x7 = Unit3D(in_shape, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2))
    #self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=get_padding())


  def forward(x):
    assert list(x.size())[1:] == in_shape[1:]
    x = self.Conv3d_1a_7x7(x)
    print(x.size())



if __name__ == '__main__':
  inp = autograd.Variable(torch.randn(1, 3, 79, 224, 224).cuda())
  model = InceptionI3d(inp.size())
  oup = model(inp)
  print(oup)
  
