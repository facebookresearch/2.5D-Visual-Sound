#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import librosa
import numpy as np
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision.transforms as transforms
import torch
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples

def main():
	#load test arguments
	opt = TestOptions().parse()
	opt.device = torch.device("cuda")

	# network builders
	builder = ModelBuilder()
	net_visual = builder.build_visual(weights=opt.weights_visual)
	net_audio = builder.build_audio(
	        ngf=opt.unet_ngf,
	        input_nc=opt.unet_input_nc,
	        output_nc=opt.unet_output_nc,
	        weights=opt.weights_audio)
	nets = (net_visual, net_audio)

	# construct our audio-visual model
	model = AudioVisualModel(nets, opt)
	model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
	model.to(opt.device)
	model.eval()

	#load the audio to perform separation
	audio, audio_rate = librosa.load(opt.input_audio_path, sr=opt.audio_sampling_rate, mono=False)
	audio_channel1 = audio[0,:]
	audio_channel2 = audio[1,:]

	#define the transformation to perform on visual frames
	vision_transform_list = [transforms.Resize((224,448)), transforms.ToTensor()]
	vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	vision_transform = transforms.Compose(vision_transform_list)

	#perform spatialization over the whole audio using a sliding window approach
	overlap_count = np.zeros((audio.shape)) #count the number of times a data point is calculated
	binaural_audio = np.zeros((audio.shape))

	#perform spatialization over the whole spectrogram in a siliding-window fashion
	sliding_window_start = 0
	data = {}
	samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
	while sliding_window_start + samples_per_window < audio.shape[-1]:
		sliding_window_end = sliding_window_start + samples_per_window
		normalizer, audio_segment = audio_normalize(audio[:,sliding_window_start:sliding_window_end])
		audio_segment_channel1 = audio_segment[0,:]
		audio_segment_channel2 = audio_segment[1,:]
		audio_segment_mix = audio_segment_channel1 + audio_segment_channel2

		data['audio_diff_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
		data['audio_mix_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
		#get the frame index for current window
		frame_index = int(round((((sliding_window_start + samples_per_window / 2.0) / audio.shape[-1]) * opt.input_audio_length + 0.05) * 10 ))
		image = Image.open(os.path.join(opt.video_frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB')
		#image = image.transpose(Image.FLIP_LEFT_RIGHT)
		frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
		data['frame'] = frame

		output = model.forward(data)
		predicted_spectrogram = output['binaural_spectrogram'][0,:,:,:].data[:].cpu().numpy()

		#ISTFT to convert back to audio
		reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
		reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
		reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
		reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
		reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

		binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
		overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
		sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

	#deal with the last segment
	normalizer, audio_segment = audio_normalize(audio[:,-samples_per_window:])
	audio_segment_channel1 = audio_segment[0,:]
	audio_segment_channel2 = audio_segment[1,:]
	data['audio_diff_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
	data['audio_mix_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
	#get the frame index for last window
	frame_index = int(round(((opt.input_audio_length - opt.audio_length / 2.0) + 0.05) * 10))
	image = Image.open(os.path.join(opt.video_frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB')
	#image = image.transpose(Image.FLIP_LEFT_RIGHT)
	frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
	data['frame'] = frame
	output = model.forward(data)
	predicted_spectrogram = output['binaural_spectrogram'][0,:,:,:].data[:].cpu().numpy()
	#ISTFT to convert back to audio
	reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
	reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
	reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
	reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
	reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

	#add the spatialized audio to reconstructed_binaural
	binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
	overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

	#divide aggregated predicted audio by their corresponding counts
	predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

	#check output directory
	if not os.path.isdir(opt.output_dir_root):
		os.mkdir(opt.output_dir_root)

	mixed_mono = (audio_channel1 + audio_channel2) / 2
	librosa.output.write_wav(os.path.join(opt.output_dir_root, 'predicted_binaural.wav'), predicted_binaural_audio, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(opt.output_dir_root, 'mixed_mono.wav'), mixed_mono, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(opt.output_dir_root, 'input_binaural.wav'), audio, opt.audio_sampling_rate)

if __name__ == '__main__':
    main()
