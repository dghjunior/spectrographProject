# This script includes the full pipline for a palatalization audio classifier.
# The code is adapted from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
# Substantial changes made to adapt model from image classification to audio classification
# Pipline is data cleanup -> feature extraction -> model building -> training -> testing

## import packages
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from skimage import io
#from skimage.io import imread, imshow
import matplotlib as plt
import random

## set directory for metadata and build wav data cleaning methods
metadata = "t_data_asr.txt"
df = pd.read_table(metadata)
data_path = "img_files/"

class img_util():
    def open(audio_file):
        sig, samplerate = torchaudio.load(audio_file)
        return (sig, samplerate)
    def feature_extraction(img_file):
        img = io.imread(img_file, as_gray=True)
        features = torch.flatten(torch.from_numpy(img))
        print(features.shape)
        return features

## create class for model input preparation
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

        img_file = self.data_path + self.df.loc[idx, 'file'] + ".png"
        class_id = self.df.loc[idx, 'coding']

        # img = img_util.open(img_file)
        img_features = img_util.feature_extraction(img_file)
        print(img_features.shape)

        return img_features, class_id

## splitting dataset into training and validation set

#load files into dataset
dataset = input_prep(df, data_path)

#split dataset
num_items = len(dataset)
num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_ds, test_ds = random_split(dataset, [num_train, num_test])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
#hpyerparamaters
input_size = 600
hidden_size_0 = 512
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









