import os
import torch
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class WeatherOption:
    def __init__(self):
        self.fog_path_1 = ('fog', '75m')
        self.fog_path_2 = ('fog', '150m')
        self.rain_path = ('mix_rain', '50mm')
        self.snow_path = ('mix_snow', 'data')
        self.raingan_path = ('raingan', 'data')
        self.snowgan_path = ('snowgan', 'data')
        self.num_domains = 6
        

class WeatherKITTLatentDataset(Dataset):
    def __init__(self, 
                 filename_ls_path='data_split/kitti_depth/eigen_train_files_with_gt.txt',
                 base_path='data/kitti/latent_train',
                 image_path='data/kitti',
                 flip_rate=0.5):
        super().__init__()
        self.weather_opt = WeatherOption()
        self.rgb_path = 'rgb'
        self.target_path = 'rgb'
        self.image_path = image_path
        self.base_path = base_path
        self.flip_rate = flip_rate
        # Load filenames
        with open(filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ] 
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]
        self.rgb_latent = [os.path.join(self.rgb_path, filename_line[0][11:].replace('.png', '.npy'))
                          for filename_line in self.filenames]
        
        self.target_latent = self.rgb_latent.copy()
        self.target_latent = self.target_latent * (self.weather_opt.num_domains + 1)
        self.get_image_file_path()
        self.get_latent_file_path()

    def get_image_file_path(self):
        self.rgb_image = [os.path.join(self.rgb_path, filename_line[0][11:])
                          for filename_line in self.filenames]
        self.rain_image = [os.path.join(self.weather_opt.rain_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.rain_path[1]))
                            for filename_line in self.filenames]
        self.raingan_image = [os.path.join(self.weather_opt.raingan_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.raingan_path[1]))
                            for filename_line in self.filenames]
        self.fog1_image = [os.path.join(self.weather_opt.fog_path_1[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_1[1]))
                            for filename_line in self.filenames]
        self.fog2_image = [os.path.join(self.weather_opt.fog_path_2[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_2[1]))
                            for filename_line in self.filenames]
        self.snow_image = [os.path.join(self.weather_opt.snow_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.snow_path[1]))
                          for filename_line in self.filenames]
        self.snowgan_image = [os.path.join(self.weather_opt.snowgan_path[0], 
                                           filename_line[0][11:].replace('data', self.weather_opt.snowgan_path[1]))
                          for filename_line in self.filenames]
        
        self.mix_image = self.rgb_image + self.rain_image + self.raingan_image + \
                                self.fog1_image + self.fog2_image + self.snow_image + self.snowgan_image

    def get_latent_file_path(self):
        self.rgb_latent = [os.path.join(self.rgb_path, filename_line[0][11:].replace('.png', '.npy'))
                          for filename_line in self.filenames]
        self.rain_latent = [os.path.join(self.weather_opt.rain_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.rain_path[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.raingan_latent = [os.path.join(self.weather_opt.raingan_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.raingan_path[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.fog1_latent = [os.path.join(self.weather_opt.fog_path_1[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_1[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.fog2_latent = [os.path.join(self.weather_opt.fog_path_2[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_2[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.snow_latent = [os.path.join(self.weather_opt.snow_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.snow_path[1]).replace('.png', '.npy'))
                          for filename_line in self.filenames]
        self.snowgan_latent = [os.path.join(self.weather_opt.snowgan_path[0], 
                                           filename_line[0][11:].replace('data', self.weather_opt.snowgan_path[1]).replace('.png', '.npy'))
                          for filename_line in self.filenames]
        self.mix_latent = self.rgb_latent + self.rain_latent + self.raingan_latent + \
                                self.fog1_latent + self.fog2_latent + self.snow_latent + self.snowgan_latent

    def _load_rgb_data(self, rgb_path, dict_names=("rgb_int", "rgb_norm")):
        image = Image.open(rgb_path)  # [H, W, 3]
        image = image.convert("RGB")
        rgb = np.asarray(image)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [3, H, W]
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_data = {
            dict_names[0]: torch.from_numpy(rgb).int(),
            dict_names[1]: torch.from_numpy(rgb_norm).float(),
        }
        rgb_data = {k: self.kitti_benchmark_crop(v) for k, v in rgb_data.items()}
        return rgb_data

    @staticmethod
    def kitti_benchmark_crop(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor: Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out

    def __len__(self):
        return len(self.mix_latent)
    
    def _get_data_path(self, index):
        latent_rel_path = self.mix_latent[index]
        target_rel_path = self.target_latent[index]
        image_rel_path = self.mix_image[index]
        return latent_rel_path, target_rel_path, image_rel_path
    
    def __getitem__(self, index):
        rgb_rel_path, target_rel_path, image_rel_path = self._get_data_path(index=index)
        sample =  {"index": index, "rgb_relative_path": rgb_rel_path}
        
        latent = np.load(os.path.join(self.base_path, rgb_rel_path))
        target = np.load(os.path.join(self.base_path, target_rel_path))
            
        latent  = torch.from_numpy(latent).squeeze()
        target = torch.from_numpy(target).squeeze()
        
        rgb_path = os.path.join(self.image_path, image_rel_path)
        rgb_data = self._load_rgb_data(rgb_path)
        rgb_int = rgb_data["rgb_int"]
        rgb_norm = rgb_data["rgb_norm"]

        if np.random.rand() < self.flip_rate:
            latent = latent.flip(dims=[-1])
            target = target.flip(dims=[-1])
            rgb_int = rgb_int.flip(dims=[-1])
            rgb_norm = rgb_norm.flip(dims=[-1])
            
        sample["input_latent"] = latent
        sample["target_latent"] = target
        sample["rgb_int"] = rgb_int
        sample["rgb_norm"] = rgb_norm
        
        return sample