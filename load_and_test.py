## import packages
# audio packages
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import random
# img packages
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import gc
import datetime
from comboclassifier import PalatalizationClassifier
from comboclassifier import inference
from comboclassifier import test_dl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PalatalizationClassifier()
model.load_state_dict(torch.load("combo_model.pt"))
model.to(device)
inference(model, test_dl)