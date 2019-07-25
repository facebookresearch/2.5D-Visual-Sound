#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio = nets

    def forward(self, input, volatile=False):
        visual_input = input['frame']
        audio_diff = input['audio_diff_spec']
        audio_mix = input['audio_mix_spec']
        audio_gt = Variable(audio_diff[:,:,:-1,:], requires_grad=False)

        input_spectrogram = Variable(audio_mix, requires_grad=False, volatile=volatile)
        visual_feature = self.net_visual(Variable(visual_input, requires_grad=False, volatile=volatile))
        mask_prediction = self.net_audio(input_spectrogram, visual_feature)

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        output =  {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_gt}
        return output
