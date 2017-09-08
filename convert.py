import pickle
import torch


rgb_pytorch = pickle.load(open('rgb_state_dict.pickle', 'rb'))
rgb_tf = pickle.load(open('rgb_variables.pickle', 'rb'))
rgb_tf = {'.'.join(k.split('/')[2:]).replace('.Branch_', '.branch')
                                    .replace('.conv_3d.', '.conv.')
                                    .replace('.Conv3d_0a_3x3', '.Conv3d_0b_3x3')
                                    .replace('.w', '.weight')
                                    .replace('conv.b', 'conv.bias')
                                    .replace('.batch_norm.', '.batchnorm.')
                                    .replace('.moving_mean', '.running_mean')
                                    .replace('.moving_variance', '.running_var')
                                    .replace('.beta', '.bias'): v for k , v in rgb_tf.items()}

assert set(rgb_tf.keys()).issubset(set(rgb_pytorch.keys()))

for k in rgb_pytorch:
  if k not in rgb_tf:
    assert k.endswith('batchnorm.weight')
    rgb_pytorch[k] = torch.ones(rgb_pytorch[k].size())
  elif k.endswith('conv.weight'):
    rgb_pytorch[k] = torch.Tensor(rgb_tf[k].transpose([4, 3, 0, 1, 2]))
  elif k.endswith('conv.bias'):
    rgb_pytorch[k] = torch.Tensor(rgb_tf[k])
  elif k.endswith('batchnorm.running_mean'):
    rgb_pytorch[k] = torch.Tensor(rgb_tf[k].squeeze())
  elif k.endswith('batchnorm.running_var'):
    rgb_pytorch[k] = torch.Tensor(rgb_tf[k].squeeze())
  elif k.endswith('batchnorm.bias'):
    rgb_pytorch[k] = torch.Tensor(rgb_tf[k].squeeze())
  else:
    raise ''

torch.save(rgb_pytorch, 'rgb_i3d.th')


flow_pytorch = pickle.load(open('flow_state_dict.pickle', 'rb'))
flow_tf = pickle.load(open('flow_variables.pickle', 'rb'))
flow_tf = {'.'.join(k.split('/')[2:]).replace('.Branch_', '.branch')
                                    .replace('.conv_3d.', '.conv.')
                                    .replace('.Conv3d_0a_3x3', '.Conv3d_0b_3x3')
                                    .replace('.w', '.weight')
                                    .replace('conv.b', 'conv.bias')
                                    .replace('.batch_norm.', '.batchnorm.')
                                    .replace('.moving_mean', '.running_mean')
                                    .replace('.moving_variance', '.running_var')
                                    .replace('.beta', '.bias'): v for k , v in flow_tf.items()}

assert set(flow_tf.keys()).issubset(set(flow_pytorch.keys()))

for k in flow_pytorch:
  if k not in flow_tf:
    assert k.endswith('batchnorm.weight')
    flow_pytorch[k] = torch.ones(flow_pytorch[k].size())
  elif k.endswith('conv.weight'):
    flow_pytorch[k] = torch.Tensor(flow_tf[k].transpose([4, 3, 0, 1, 2]))
  elif k.endswith('conv.bias'):
    flow_pytorch[k] = torch.Tensor(flow_tf[k])
  elif k.endswith('batchnorm.running_mean'):
    flow_pytorch[k] = torch.Tensor(flow_tf[k].squeeze())
  elif k.endswith('batchnorm.running_var'):
    flow_pytorch[k] = torch.Tensor(flow_tf[k].squeeze())
  elif k.endswith('batchnorm.bias'):
    flow_pytorch[k] = torch.Tensor(flow_tf[k].squeeze())
  else:
    raise ''

torch.save(flow_pytorch, 'flow_i3d.th') 

