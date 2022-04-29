from ast import increment_lineno
from random import sample
import os
from numpy import maximum_sctype
from scipy.io import wavfile
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import random

# Source for spectrograph generation https://pythontic.com/visualization/signals/spectrogram
# Also: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

# Model building guides from Austin:

# https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

def create_spectrogram(filename=''):
    if filename != '':
        sampling_frequency, signal_data = wavfile.read(filename)

        fig = plt.figure()
        plt.title(filename[39:])

        plt.specgram(signal_data, Fs=sampling_frequency)
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        index = filename.find('__')
        index2 = filename.rfind('__')
        classifier = 'img_files/' + filename[index:index2]
        if not os.path.exists(classifier):
            os.makedirs(classifier)
        img_filename = classifier + '/' + filename[index:len(filename)-3] + 'png'
        print(img_filename)
        fig.savefig(img_filename)
        plt.close(fig)

def spectrograph_loop():
    directory = 'prepped_wav_files'
    i = 0
    for file in os.listdir(directory):
        i += 1
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            print(str(i), end=', ')
            create_spectrogram(f)

def trim_pad_audio():
    directory = 'wav_files'
    i = 0
    for file in os.listdir(directory):
        i += 1
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            sig, samplerate = torchaudio.load(f)
            num_rows, sig_len = sig.shape
            max_len = samplerate // 1000 * 700
            if sig_len > max_len:
                sig = sig[:,:max_len]
            elif sig_len < max_len:
                pad_begin_len = random.randint(0, max_len - sig_len)
                pad_end_len = max_len - sig_len - pad_begin_len
                pad_begin = torch.zeros((num_rows, pad_begin_len))
                pad_end = torch.zeros((num_rows, pad_end_len))
                sig = torch.cat((sig, pad_begin, pad_end), 1)
            filename = 'prepped_wav_files/' + file
            print(str(i) + ', ' + filename)
            torchaudio.save(filename, sig, samplerate)

#spectrograph_loop()
#create_spectrogram('prepped_wav_files/021A-C0897X0004XX-AAZZP0_000407_KDP_2__WHAT-YOU__1675-3025.wav')
#trim_pad_audio()

if not os.path.exists('img_files'):
    os.makedirs('img_files')
data_dir = 'img_files'

# Create training and validation data loaders
train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
train_ds = datasets.ImageFolder(data_dir, transform=train_transforms)
test_ds = datasets.ImageFolder(data_dir, transform=test_transforms)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
#hpyerparamaters
input_size =741
hidden_size_0 = 250
hidden_size_1 = 100
num_classes = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## create feed forward network
class PalatalizationClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_0, num_classes):
        super(PalatalizationClassifier, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size_0)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_0, hidden_size_1)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size_1, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

##training loop
def training(model, train_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0.0
        total_prediction = 0.0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs,1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_baches = len(train_dl)
        avg_loss = running_loss/num_baches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('finished')

## Inference fucntion
def inference(model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    #disabling gradient updates
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

## Instantiate model on GPU
PalatalizationClassifier = PalatalizationClassifier(input_size, hidden_size_0, num_classes)
PalatalizationClassifier = PalatalizationClassifier.to(device)
## Training
num_epochs=50
training(PalatalizationClassifier, train_dl, num_epochs)

## Testing
inference(PalatalizationClassifier, test_dl)