# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from model import InceptionI3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb_imagenet': 'rgb_i3d.th',
    'flow_imagenet': 'flow_i3d.th',
}

_LABEL_MAP_PATH = 'data/label_map.txt'


def main():
  eval_type = 'joint'
  imagenet_pretrained = True

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  rgb_model = InceptionI3d(3)
  rgb_model.load_state_dict(torch.load(_CHECKPOINT_PATHS['rgb_imagenet']))
  rgb_model.cuda()
  rgb_model.eval()
  print('RGB checkpoint restored')

  flow_model = InceptionI3d(2)
  flow_model.load_state_dict(torch.load(_CHECKPOINT_PATHS['flow_imagenet']))
  flow_model.cuda()
  flow_model.eval()
  print('FLOW checkpoint restored')

  rgb_sample = np.load(_SAMPLE_PATHS['rgb']).transpose([0, 4, 1, 2, 3])
  print('RGB data loaded, shape=%s', str(rgb_sample.shape))
  flow_sample = np.load(_SAMPLE_PATHS['flow']).transpose([0, 4, 1, 2, 3])
  print('Flow data loaded, shape=%s', str(flow_sample.shape))
  
  rgb_sample = autograd.Variable(torch.Tensor(rgb_sample).cuda())
  rgb_logits = rgb_model(rgb_sample)
  flow_sample = autograd.Variable(torch.Tensor(flow_sample).cuda())
  flow_logits = flow_model(flow_sample)
  model_logits = rgb_logits + flow_logits
  model_predictions = F.softmax(model_logits)
  
  out_logits = model_logits.data.cpu().numpy()[0]
  out_predictions = model_predictions.data.cpu().numpy()[0]
  sorted_indices = np.argsort(out_predictions)[::-1]

  print('Norm of logits: %f' % np.linalg.norm(out_logits))
  print('\nTop classes and probabilities')
  for index in sorted_indices[:20]:
    print(out_predictions[index], out_logits[index], kinetics_classes[index])


if __name__ == '__main__':
  main()
