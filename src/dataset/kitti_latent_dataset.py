import os
import torch
import random
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
                 base_path = 'data/kitti/latent_train',
                 flip_rate=0.5):
        super().__init__()
        self.weather_opt = WeatherOption()
        self.rgb_path = 'rgb'
        self.target_path = 'rgb'
        self.base_path = base_path
        self.flip_rate = flip_rate
        # Load filenames
        with open(filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ] 
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]
        
        self.rgb_files = [os.path.join(self.rgb_path, filename_line[0][11:].replace('.png', '.npy'))
                          for filename_line in self.filenames]
        
        self.target_files = self.rgb_files.copy()
        
        self.rain_files = [os.path.join(self.weather_opt.rain_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.rain_path[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.raingan_files = [os.path.join(self.weather_opt.raingan_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.raingan_path[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.fog1_files = [os.path.join(self.weather_opt.fog_path_1[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_1[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.fog2_files = [os.path.join(self.weather_opt.fog_path_2[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_2[1]).replace('.png', '.npy'))
                            for filename_line in self.filenames]
        self.snow_files = [os.path.join(self.weather_opt.snow_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.snow_path[1]).replace('.png', '.npy'))
                          for filename_line in self.filenames]
        self.snowgan_files = [os.path.join(self.weather_opt.snowgan_path[0], 
                                           filename_line[0][11:].replace('data', self.weather_opt.snowgan_path[1]).replace('.png', '.npy'))
                          for filename_line in self.filenames]
        
        self.mix_filenames = self.rgb_files + self.rain_files + self.raingan_files + \
                                self.fog1_files + self.fog2_files + self.snow_files + self.snowgan_files

        self.target_files = self.target_files * (self.weather_opt.num_domains + 1)

    def __len__(self):
        return len(self.mix_filenames)
    
    def _get_data_path(self, index):
        rgb_rel_path = self.mix_filenames[index]
        target_rel_path = self.target_files[index]
        
        return rgb_rel_path, target_rel_path
    
    def __getitem__(self, index):
        rgb_rel_path, target_rel_path = self._get_data_path(index=index)
        sample =  {"index": index, "rgb_relative_path": rgb_rel_path}
        
        latent = np.load(os.path.join(self.base_path, rgb_rel_path))
        target = np.load(os.path.join(self.base_path, target_rel_path))
            
        latent  = torch.from_numpy(latent).squeeze()
        target = torch.from_numpy(target).squeeze()
        
        if np.random.rand() < self.flip_rate:
            latent = latent.flip(dims=[-1])
            target = target.flip(dims=[-1])
            
        sample["input_latent"] = latent
        sample["target_latent"] = target
        
        return sample