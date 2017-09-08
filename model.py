import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def get_padding(in_duration, in_height, in_width, kernel_size, stride):
    out_duration = math.ceil(in_duration / float(stride[0]))
    out_height = math.ceil(in_height / float(stride[1]))
    out_width  = math.ceil(in_width / float(stride[2]))

    if in_duration % stride[0] == 0:
      pad_duration = max(kernel_size[0] - stride[0], 0) // 2
    else:
      pad_duration = max(kernel_size[0] - (in_duration % stride[0]), 0) // 2
    if in_height % stride[1] == 0:
      pad_height = max(kernel_size[1] - stride[1], 0) // 2
    else:
      pad_height = max(kernel_size[1] - (in_height % stride[1]), 0) // 2
    if in_width % stride[2] == 0:
      pad_width = max(kernel_size[2] - stride[2], 0) // 2
    else:
      pad_width = max(kernel_size[2] - (in_width % stride[2]), 0) // 2

    return (pad_duration, pad_height, pad_width)



class Unit3D(nn.Module):
  
  def __init__(self, in_channels,
               out_channels, 
               kernel_size=(1, 1, 1), 
               stride=(1, 1, 1),
               padding=(0, 0, 0),
               activation_fn=F.relu, 
               batch_norm=True, 
               bias=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.activation_fn = activation_fn
    self.batch_norm = batch_norm
    self.bias = bias

    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                          stride, padding, bias=self.bias)
    if self.batch_norm:
      self.batchnorm = nn.BatchNorm3d(out_channels)

 
  def forward(self, x):
    x = self.conv(x)
    if self.batch_norm:
      x = self.batchnorm(x)
    if self.activation_fn != None:
      x = self.activation_fn(x)
    return x


class MixedBlock(nn.Module):

  def __init__(self, in_duration, in_height, in_width, in_channels, output_channels):
    super().__init__()
    self.branch0 = nn.Sequential(OrderedDict([
      ('Conv3d_0a_1x1', Unit3D(in_channels, output_channels[0][0])),
    ]))
    self.branch1 = nn.Sequential(OrderedDict([
      ('Conv3d_0a_1x1', Unit3D(in_channels, output_channels[1][0])),
      ('Conv3d_0b_3x3', Unit3D(output_channels[1][0], output_channels[1][1], 
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))),
    ]))
    self.branch2 = nn.Sequential(OrderedDict([
      ('Conv3d_0a_1x1', Unit3D(in_channels, output_channels[2][0])),
      ('Conv3d_0b_3x3', Unit3D(output_channels[2][0], output_channels[2][1], 
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))),
    ]))
    self.branch3 = nn.Sequential(OrderedDict([
      ('MaxPool3d_0a_3x3', nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), 
                                        padding=get_padding(in_duration, in_height, in_width, (3, 3, 3), (1, 1, 1)))),
      ('Conv3d_0b_1x1', Unit3D(in_channels, output_channels[3][0])),
    ]))


  def forward(self, x):
    return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1)



class InceptionI3d(nn.Module):

  def __init__(self, in_channels, num_classes=400, dropout_prob=0.):
    super().__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes

    self.Conv3d_1a_7x7 = Unit3D(in_channels, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), 
                                padding=get_padding(79, 224, 224, (7, 7, 7), (2, 2, 2)))
    self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), 
                                         padding=get_padding(40, 111, 111, (1, 3, 3), (1, 2, 2)))
    self.Conv3d_2b_1x1 = Unit3D(in_channels=64, out_channels=64,
                                padding=get_padding(40, 56, 56, (1, 1, 1), (1, 1, 1)))
    self.Conv3d_2c_3x3 = Unit3D(in_channels=64, out_channels=192, kernel_size=(3, 3, 3),
                                padding=get_padding(40, 56, 56, (3, 3, 3), (1, 1, 1)))
    self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=get_padding(40, 111, 111, (1, 3, 3), (1, 2, 2)))
    self.Mixed_3b = MixedBlock(40, 28, 28, 192, [[64], [96, 128], [16, 32], [32]])
    self.Mixed_3c = MixedBlock(40, 28, 28, 256, [[128], [128, 192], [32, 96], [64]])
    self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    self.Mixed_4b = MixedBlock(20, 14, 14, 480, [[192], [96, 208], [16, 48], [64]])
    self.Mixed_4c = MixedBlock(20, 14, 14, 512, [[160], [112, 224], [24, 64], [64]]) 
    self.Mixed_4d = MixedBlock(20, 14, 14, 512, [[128], [128, 256], [24, 64], [64]])
    self.Mixed_4e = MixedBlock(20, 14, 14, 512, [[112], [144, 288], [32, 64], [64]])
    self.Mixed_4f = MixedBlock(20, 14, 14, 528, [[256], [160, 320], [32, 128], [128]])
    self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
    self.Mixed_5b = MixedBlock(10, 7, 7, 832, [[256], [160, 320], [32, 128], [128]])
    self.Mixed_5c = MixedBlock(10, 7, 7, 832, [[384], [192, 384], [48, 128], [128]])
    self.Logits = nn.Sequential(OrderedDict([
      ('avg_pool', nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))),
      ('dropout', nn.Dropout(dropout_prob)),
      ('Conv3d_0c_1x1', Unit3D(in_channels=1024, out_channels=num_classes, kernel_size=(1, 1, 1),
                               activation_fn=None, batch_norm=None, bias=True)),
    ]))



  def forward(self, x):
    assert list(x.size())[1:] == [self.in_channels, 79, 224, 224]
 
    x = self.Conv3d_1a_7x7(x)
    x = self.MaxPool3d_2a_3x3(x)
    x = self.Conv3d_2b_1x1(x)
    x = self.Conv3d_2c_3x3(x)
    x = self.MaxPool3d_3a_3x3(x)
    x = self.Mixed_3b(x)
    x = self.Mixed_3c(x)
    x = self.MaxPool3d_4a_3x3(x)
    x = self.Mixed_4b(x)
    x = self.Mixed_4c(x)
    x = self.Mixed_4d(x)
    x = self.Mixed_4e(x)
    x = self.Mixed_4f(x)
    x = self.MaxPool3d_5a_2x2(x)
    x = self.Mixed_5b(x)
    x = self.Mixed_5c(x)
    x = self.Logits(x)
    logits = torch.squeeze(torch.squeeze(x, 3), 3)
    avg_logits = torch.mean(logits, 2)

    return avg_logits


if __name__ == '__main__':
  inp = autograd.Variable(torch.randn(1, 2, 79, 224, 224).cuda())

  model = InceptionI3d(2)
  import pickle
  pickle.dump(model.state_dict(), open('model_state_dict.pickle', 'wb'))
  model.cuda()

  oup = model(inp)
  print(oup)
  
