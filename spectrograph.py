import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch
import torchaudio
import random

# Source for spectrograph generation https://pythontic.com/visualization/signals/spectrogram
# Also: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

def create_spectrogram(filename='', sorted = False):
    if filename != '':
        sampling_frequency, signal_data = wavfile.read(filename)
        plt.figure()
        plt.title(filename)
        plt.specgram(signal_data, Fs=sampling_frequency)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.axis('off')
        if sorted:
            index = filename.find('__')
            index2 = filename.rfind('__')
            classifier = 'sorted_img_files/' + filename[index+2:index2]
            if not os.path.exists(classifier):
                os.makedirs(classifier)
            img_filename = classifier + '/' + filename[index+2:len(filename)-3] + 'png'
        else:
            img_filename = 'NEW/' + filename[17:len(filename)-3] + 'png'
        print(img_filename)
        #plt.savefig(img_filename, bbox_inches='tight')
        plt.savefig(img_filename)
        plt.close()

def spectrograph_loop(sorted = False):
    if not os.path.exists('NEW'):
        os.makedirs('NEW')
    directory = 'prepped_mini_set'
    i = 0
    for file in os.listdir(directory):
        i += 1
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            print(str(i), end=', ')
            create_spectrogram(f, sorted)

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

spectrograph_loop(False)