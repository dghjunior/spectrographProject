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

class audio_util():
    #read wav file
    def open(audio_file):
        sig, samplerate = torchaudio.load(audio_file)
        return (sig, samplerate)
    #trim or pad audio to chosen max size
    def pad_trunk(audio, max_ms):
        sig, samplerate = audio
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

        return (sig, samplerate)

class input_prep(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 700
        self.sr = 16000
        self.channel = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        audio_file = self.data_path + self.df.loc[idx, 'file'] + ".wav"
        class_id = self.df.loc[idx, 'coding']

        audio = audio_util.open(audio_file)
        padded_audio = audio_util.pad_trunk(audio, self.duration)
        return padded_audio

def createSpectrogram(filename=''):
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

def new(filename = ''):
    if filename != '':
        samplingFrequency, signalData = wavfile.read(filename)

        fig, ax = plt.subplots()
        ax.set_title(filename[48:])
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 8000)
        plt.specgram(signalData, Fs=samplingFrequency)
        imgFilename = 'imgFiles/' + filename[9:len(filename)-3] + 'png'
        print(imgFilename)
        fig.savefig(imgFilename)

def spectrograph_loop():
    directory = 'wavFiles'
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            #createSpectrogram(f)
            new(f)

# spectrograph_loop()
new('wavFiles/021A-C0897X0004XX-AAZZP0_000407_KDP_2__WHAT-YOU__1675-3025.wav')

# Model building guides from Austin:

# https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5