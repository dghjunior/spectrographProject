from random import sample

import matplotlib.pyplot as plt
import os
from scipy.io import wavfile

# Source for spectrograph generation https://pythontic.com/visualization/signals/spectrogram

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

directory = 'wavFiles'
# os.remove('imgFiles')
# os.mkdir('imgFiles')
for file in os.listdir(directory):
    f = os.path.join(directory, file)
    if os.path.isfile(f):
        createSpectrogram(f)

# Model building guides from Austin:

# https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5