#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import glob
import argparse
import numpy as np
import resampy
from scikits.audiolab import Sndfile, Format

def load_wav(fname, rate=None):
    fp = Sndfile(fname, 'r')
    _signal = fp.read_frames(fp.nframes)
    _signal = _signal.reshape((-1, fp.channels))
    _rate = fp.samplerate

    if _signal.ndim == 1:
        _signal.reshape((-1, 1))
    if rate is not None and rate != _rate:
        signal = resampy.resample(_signal, _rate, rate, axis=0, filter='kaiser_best')
    else:
        signal = _signal
        rate = _rate

    return signal, rate

def save_wav(fname, signal, rate):
    fp = Sndfile(fname, 'w', Format('wav'), signal.shape[1], rate)
    fp.write_frames(signal)
    fp.close()

def reEncodeAudio(audio_path, new_rate):
	audio, audio_rate = load_wav(audio_path,new_rate)
	save_wav(audio_path, audio, new_rate)

def main():
	parser = argparse.ArgumentParser(description="re-encode all audios under a directory")
	parser.add_argument("--audio_dir_path", type=str, required=True)
	parser.add_argument("--new_rate", type=int, default=16000)
	args = parser.parse_args()

	audio_list = glob.glob(args.audio_dir_path + '/*.wav')
	print "Total number of audios to re-encode: ", len(audio_list)
	for audio_path in audio_list:
		reEncodeAudio(os.path.join(args.audio_dir_path, audio_path), args.new_rate)

if __name__ == '__main__':
	main()
