import random
from os.path import join
import numpy as np
import torch
import constants as c
from loader import Loader
from model import GenModel
from utils import list_paths, calculate_psnr, var2np, makedir
from scipy.misc import imsave
from loader import float32_to_uint8


class Runner:
    def __init__(self):

        if c.SEED:
            torch.manual_seed(c.SEED)
            np.random.seed(c.SEED)
            random.seed(c.SEED)

        self.g_model = GenModel()
        if c.LOAD_GEN:
            self.load_model()

    def process_video(self, loader, save_images=True):

        psnr_all = []

        while not loader.is_done():

            input_sequence, gt, gt_path = loader.get_sequence()
            gen = self.g_model(input_sequence)
            psnr_frame = calculate_psnr(gen, gt, val_range=2)
            psnr_all.append(psnr_frame)

            if save_images:
                gen_folder = makedir(join(c.IMG_SAVE_DIR, gt_path.split('/')[-2]))
                gen_path = join(gen_folder, gt_path.split('/')[-1])
                gen_uint8 = float32_to_uint8(gen)
                imsave(gen_path, var2np(gen_uint8))

        psnr_mean = np.mean(psnr_all)

        return psnr_mean

    def test(self):

        with torch.no_grad():

            self.g_model.eval()
            test_videos = list_paths(c.TEST_DIR)
            psnr_all_videos = []

            for vid in test_videos:  # test videos seperately
                vid_name = vid.split('/')[-1]
                video_loader = Loader(folder=vid)
                psnr_mean = self.process_video(video_loader)
                print('{:15s} - PSNR: {:5.2f}'.format(vid_name, psnr_mean))
                psnr_all_videos.append(psnr_mean)

            psnr_videos_mean = np.mean(psnr_all_videos)
            print('{:15s} - PSNR: {:5.2f}'.format('AVERAGE', psnr_videos_mean))
            print('\nPredicted frames are saved at {}'.format(c.IMG_SAVE_DIR))

    def load_model(self):

        file_path = join(c.MODEL_DIR, c.LOAD_GEN + '.pt')
        data = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.g_model.load_state_dict(data['g_model'])
        print('Generator restored\n')


def main():

    runner = Runner()
    runner.test()


if __name__ == '__main__':

    main()
