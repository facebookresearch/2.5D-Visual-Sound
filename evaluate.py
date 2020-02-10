#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import librosa
import argparse
import numpy as np
from numpy import linalg as LA
from scipy.signal import hilbert
from data.audioVisual_dataset import generate_spectrogram
import statistics as stat

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def STFT_L2_distance(predicted_binaural, gt_binaural):
    #channel1
    predicted_spect_channel1 = librosa.core.stft(np.asfortranarray(predicted_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    gt_spect_channel1 = librosa.core.stft(np.asfortranarray(gt_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    #channel2
    predicted_spect_channel2 = librosa.core.stft(np.asfortranarray(predicted_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    gt_spect_channel2 = librosa.core.stft(np.asfortranarray(gt_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    #sum the distance between two channels
    stft_l2_distance = channel1_distance + channel2_distance
    return float(stft_l2_distance)

def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_root', type=str, required=True)
	parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
	parser.add_argument('--real_mono', default=False, type=bool, help='whether the input predicted binaural audio is mono audio')
	parser.add_argument('--normalization', default=False, type=bool)
	args = parser.parse_args()
	stft_distance_list = []
	envelope_distance_list = []

	audioNames = os.listdir(args.results_root)
	index = 1
	for audio_name in audioNames:
		if index % 10 == 0:
			print "Evaluating testing example " + str(index) + " :", audio_name
		#check whether input binaural is mono, replicate to two channels if it's mono
		if args.real_mono:
			mono_sound, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'mixed_mono.wav'), sr=args.audio_sampling_rate)
			predicted_binaural = np.repeat(np.expand_dims(mono_sound, 0), 2, axis=0)
			if args.normalization:
				predicted_binaural = normalize(predicted_binaural)
		else:
			predicted_binaural, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'predicted_binaural.wav'), sr=args.audio_sampling_rate, mono=False)
			if args.normalization:
				predicted_binaural = normalize(predicted_binaural)
		gt_binaural, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'input_binaural.wav'), sr=args.audio_sampling_rate, mono=False)
		if args.normalization:
			gt_binaural = normalize(gt_binaural)
		
		#get results for this audio
		stft_distance_list.append(STFT_L2_distance(predicted_binaural, gt_binaural))
		envelope_distance_list.append(Envelope_distance(predicted_binaural, gt_binaural))
		index = index + 1

	#print the results
	print "STFT L2 Distance: ", stat.mean(stft_distance_list), stat.stdev(stft_distance_list), stat.stdev(stft_distance_list) / np.sqrt(len(stft_distance_list))
	print "Average Envelope Distance: ", stat.mean(envelope_distance_list), stat.stdev(envelope_distance_list), stat.stdev(envelope_distance_list) / np.sqrt(len(envelope_distance_list))

if __name__ == '__main__':
	main()
