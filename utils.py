import os
from math import log
import numpy as np
import torch
import constants as c
import matplotlib.pyplot as plt


def makedir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def list_paths(folder, path=True):
    output = []
    if isinstance(folder, list):
        for single_folder in folder:
            if path:
                output += sorted([os.path.join(single_folder, file_) for file_ in os.listdir(single_folder)])
            else:
                output += sorted(os.listdir(single_folder))
    else:
        if path:
            output = sorted([os.path.join(folder, file_) for file_ in os.listdir(folder)])
        else:
            output = sorted(os.listdir(folder))

    return output


def log10(x):
    return torch.log(x)/log(10)


def calculate_psnr(gen_frames, gt_frames, val_range=None, seperate=False):

    if val_range is None:
        val_range = c.MAX - c.MIN

    if type(gen_frames).__name__ == 'Tensor':
        gt_frames = gt_frames.float()
        gen_frames = gen_frames.float()

        mse = torch.mean((gt_frames - gen_frames)**2)
        psnr = 10 * log10(val_range ** 2 / mse)
        return var2np(psnr)

    elif type(gen_frames).__name__ == 'ndarray':

        gt_frames = gt_frames.astype(np.float32)
        gen_frames = gen_frames.astype(np.float32)

        mse = np.mean(((gt_frames - gen_frames)**2))
        psnr = 10 * np.log10(val_range ** 2 / mse)
        return psnr


def var2np(x, keepdims=False):
    if type(x).__name__ != 'float' and type(x).__name__ != 'float64':
        if type(x).__name__ == 'list':
            x = torch.stack(x)
        if x.shape == ():
            x = x.detach().cpu().item()
        else:
            x = x.detach().cpu().numpy()
    if not keepdims:
        x = np.squeeze(x)
    return x


def plot_prediction(video_name, psnr_all):
    frames = range(c.HIST_LEN + 1, c.HIST_LEN + 1 + len(psnr_all))
    plt.plot(frames, psnr_all, c='r')
    plt.title(video_name)
    plt.xlabel('Frame #')
    plt.ylabel('Prediction PSNR (dB)')
    plt.grid()
    plt.savefig(os.path.join(c.FIG_SAVE_DIR, video_name), dpi=150)
    plt.clf()
