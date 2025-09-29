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


class WeatherKITTILatentSceneDataset(WeatherKITTLatentDataset):
    def __init__(self, 
                 filename_ls_path='data_split/kitti_depth/eigen_train_files_with_gt.txt', 
                 base_path='data/kitti/latent_train', 
                 image_path='data/kitti', 
                 flip_rate=0.5):
        super().__init__(filename_ls_path, base_path, image_path, flip_rate)
        self.domains = ['sunny', 'rain', 'raingan', 'snow', 'snowgan', 'fog1', 'fog2']
        
    def __len__(self):
        return len(self.rgb_latent)
    
    def _get_data_path(self, index):
        # Sunny domain (Original KITTI)
        sunny_image_path = self.rgb_image[index]
        sunny_latent_path = self.rgb_latent[index]
        
        # Weather domain (Weather KITTI)
        rain_image_path = self.rain_image[index]
        raingan_image_path = self.raingan_image[index]
        snow_image_path = self.snow_image[index]
        snowgan_image_path = self.snowgan_image[index]
        fog1_image_path = self.fog1_image[index]
        fog2_image_path = self.fog2_image[index]
        
        rain_latent_path = self.rain_latent[index]
        raingan_latent_path = self.raingan_latent[index]
        snow_latent_path = self.snow_latent[index]
        snowgan_latent_path = self.snowgan_latent[index]
        fog1_latent_path = self.fog1_latent[index]
        fog2_latent_path = self.fog2_latent[index]
        
        return {'sunny': (sunny_image_path, sunny_latent_path),
                'rain': (rain_image_path, rain_latent_path),
                'raingan': (raingan_image_path, raingan_latent_path),
                'snow': (snow_image_path, snow_latent_path),
                'snowgan': (snowgan_image_path, snowgan_latent_path),
                'fog1': (fog1_image_path, fog1_latent_path),
                'fog2': (fog2_image_path, fog2_latent_path),
                }
    
    def __getitem__(self, index):
        path_dict = self._get_data_path(index=index)
        ids = torch.tensor(index, dtype=torch.int64)
        sample =  {"index": index, "rgb_relative_path": path_dict['sunny'][0]}

        sample['weather'] = []
        flip_flag = True if np.random.rand() < self.flip_rate else False
            
        for domain in self.domains:
            image = self._load_rgb_data(os.path.join(self.image_path, path_dict[domain][0]))["rgb_norm"]
            latent = np.load(os.path.join(self.base_path, path_dict[domain][1]))
            latent = torch.from_numpy(latent).squeeze()
            if domain == 'sunny':
                if flip_flag:
                    sample['sunny'] = image.flip(dims=[-1])
                else:
                    sample['sunny'] = image
            else:
                if flip_flag:
                    sample['weather'].append(image.flip(dims=[-1]))
                else:
                    sample['weather'].append(image)

        return sample


if __name__ == '__main__':
    dataset = WeatherKITTILatentSceneDataset()
    print(f"Total images: {len(dataset)}")
    
    # Example
    item = dataset[20]
    sunny = item['sunny']
    weather_list = item['weather']
    
    print("Sunny: ", sunny.shape)
    print(item['index'])
    for weather in weather_list:
        print(weather[0].shape)
    