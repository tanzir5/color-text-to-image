import argparse
import sys
print(sys.executable)
print(sys.version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from skimage import color
from sklearn.metrics import r2_score
from transformers import RobertaTokenizer, RobertaModel
import copy
from math import inf
import logging
from tqdm import tqdm
import sys

def compute_ciede2000(y_true, y_pred):
    """
    Compute CIEDE2000 color difference per predicted color.
    Both inputs are assumed to be in [0, 255] range.
    """
    y_true = np.array(y_true) / 255.0
    y_pred = np.array(y_pred) / 255.0
    y_true_lab = color.rgb2lab(y_true.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred.reshape(-1, 1, 3)).reshape(-1, 3)
    return color.deltaE_ciede2000(y_true_lab, y_pred_lab)

def compute_metrics(targets, predictions):

    # rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    r2 = r2_score(targets, predictions)

    return mae, ciede2000, r2

if __name__ == '__main__':
    predicted_path = sys.argv[1]
    model = sys.argv[2]
    rm, gm, bm = f'r_{model}', f'g_{model}', f'b_{model}'
    df = pd.read_csv(predicted_path, usecols = ['r', 'g', 'b', rm, gm, bm])
    yt = df[['r', 'g', 'b']].values
    yp = df[[rm, gm, bm]].values
    mae, ciede, r2 = compute_metrics(yt, yp)
    print(mae, ciede, r2)



