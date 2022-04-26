from random import sample

import matplotlib.pyplot as plt
import os
from numpy import maximum_sctype
from scipy.io import wavfile

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import random

# Source for spectrograph generation https://pythontic.com/visualization/signals/spectrogram
# Also: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

def oldcreateSpectrogram(filename=''):
    if filename != '':
        samplingFrequency, signalData = wavfile.read(filename)

        fig = plt.figure()
        plt.title(filename[39:])

        plt.specgram(signalData, Fs=samplingFrequency)
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        # plt.show()
        imgFilename = 'imgFiles/' + filename[9:len(filename)-3] + 'png'
        print(imgFilename)
        fig.savefig(imgFilename)

def createSpectrogram(filename=''):
    max_ms = 7
    if filename != '':
        sig, samplerate = torchaudio.load(filename)
        num_rows, sig_len = sig.shape
        max_len = samplerate // 1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((sig, pad_begin, pad_end), 1)

        waveform = sig.numpy()
        xlim = None
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / samplerate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs = samplerate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle('Spectrogram')
        plt.show(block = False)
        '''
        fig = plt.figure()
        plt.title(filename[39:])
        plt.specgram(sig, Fs=samplerate)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        img_filename = 'imgFiles/' + filename[9:len(filename)-3] + 'png'
        print(img_filename)
        fig.savefig(img_filename)
        '''

        

directory = 'wavFiles'
for file in os.listdir(directory):
    f = os.path.join(directory, file)
    if os.path.isfile(f):
        createSpectrogram(f)

# Model building guides from Austin:

# https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5