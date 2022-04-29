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

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler = train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler = test_sampler, batch_size=64)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, 0.2)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
print(model)

for param in model.parameters():
    param.required_grad = False
model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(), nn.Linear(512, 10), nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 3
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'aerialmodel.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()