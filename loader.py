from copy import deepcopy
from scipy.ndimage import imread
import torch
import constants as c
from utils import list_paths


def float32_to_uint8(data):
    data = unnormalize(data)  # to 0-255
    data = torch.round(data)  # round values
    data = data.byte()  # to uint8
    return data


def uint8_to_float32(data):
    data = data.float()  # to float
    data = normalize(data)  # to -1-1
    return data


def normalize(x):
    return x / 255.0 * (c.MAX - c.MIN) + c.MIN


def unnormalize(x):
    return (x - c.MIN) / (c.MAX - c.MIN) * 255.0


def var_normalize(x, in_range, out_range):
    in_min, in_max = in_range
    out_min, out_max = out_range
    return (x - in_min) * 1.0 / (in_max - in_min) * (out_max - out_min) + out_min


def cpu2gpu(batch):
    batch = torch.from_numpy(batch)
    batch.requires_grad = False
    if c.CUDA:
        batch = batch.cuda()
    return batch


class Loader:

    def __init__(self, folder):

        self.frames_list = list_paths(folder)
        self.ind = 0
        self.done = False
        self.first = True
        self.sequence_list = []

    def get_image(self):

        path = self.frames_list[self.ind]
        frame = imread(path, mode='L')
        frame = cpu2gpu(frame)
        frame = uint8_to_float32(frame)
        return frame, path

    def get_sequence(self):

        if self.first:
            for i in range(c.SEQ_LEN):
                frame, path = self.get_image()
                self.sequence_list.append(frame)
                self.ind += 1
            self.first = False

        else:
            frame, path = self.get_image()
            self.sequence_list.pop(0)
            self.sequence_list.append(frame)
            self.ind += 1

        sequence_tensor = torch.unsqueeze(torch.stack(self.sequence_list), dim=0)
        input_sequence = sequence_tensor[:, :c.HIST_LEN, :, :]
        gt = sequence_tensor[:, c.HIST_LEN:, :, :]
        gt_path = deepcopy(path)

        return input_sequence, gt, gt_path

    def is_done(self):
        return self.ind >= len(self.frames_list)

