# --------------------------------------------------------------------------
# Modified from Marigold:
import os
import torch
from .base_depth_dataset import \
    BaseDepthDataset, DepthFileNameMode, DatasetMode


class WeatherOption:
    def __init__(self):
        self.fog_path_1 = ('fog', '75m')
        self.fog_path_2 = ('fog', '150m')
        self.rain_path = ('mix_rain', '50mm')
        self.snow_path = ('mix_snow', 'data')
        self.raingan_path = ('raingan', 'data')
        self.snowgan_path = ('snowgan', 'data')
        self.num_domains = 6
        

class WeathewrKITTIDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        kitti_bm_crop,  # Crop to KITTI benchmark size
        valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
            # KITTI data parameter
            min_depth=1e-5,
            max_depth=80,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,)
        
        self.weather_opt = WeatherOption()
        self.rgb_path = 'rgb'
        self.depth_path = 'depth'
        
        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        assert self.valid_mask_crop in [
            None,
            "garg",   # set evaluation mask according to Garg ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]

        self.rgb_files = [os.path.join(self.rgb_path, filename_line[0][11:])
                          for filename_line in self.filenames]
        self.depth_files = [os.path.join(self.depth_path, filename_line[1])
                          for filename_line in self.filenames] * (self.weather_opt.num_domains + 1)
        self.filled = [os.path.join(self.depth_path, filename_line[2])
                          for filename_line in self.filenames] * (self.weather_opt.num_domains + 1)
        
        self.rain_files = [os.path.join(self.weather_opt.rain_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.rain_path[1]))
                            for filename_line in self.filenames]
        self.raingan_files = [os.path.join(self.weather_opt.raingan_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.raingan_path[1]))
                            for filename_line in self.filenames]
        self.fog1_files = [os.path.join(self.weather_opt.fog_path_1[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_1[1]))
                            for filename_line in self.filenames]
        self.fog2_files = [os.path.join(self.weather_opt.fog_path_2[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_2[1]))
                            for filename_line in self.filenames]
        self.snow_files = [os.path.join(self.weather_opt.snow_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.snow_path[1]))
                          for filename_line in self.filenames]
        self.snowgan_files = [os.path.join(self.weather_opt.snowgan_path[0], 
                                           filename_line[0][11:].replace('data', self.weather_opt.snowgan_path[1]))
                          for filename_line in self.filenames]
        
    def __len__(self):
        if self.version == 'mixed':
            return len(self.filenames) * self.weather_opt.num_domains
        else:
            return len(self.filenames)
        
    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode KITTI depth
        depth_decoded = depth_in / 256.0
        return depth_decoded

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        if self.kitti_bm_crop:
            rgb_data = {k: self.kitti_benchmark_crop(v) for k, v in rgb_data.items()}
        return rgb_data

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        depth_data = super()._load_depth_data(depth_rel_path, filled_rel_path)
        if self.kitti_bm_crop:
            depth_data = {
                k: self.kitti_benchmark_crop(v) for k, v in depth_data.items()
            }
        return depth_data
    
    # def _get_data_path(self, index):
    #     filename_line = self.filenames[index]
    #     # e.g., rgb/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
    #     rgb_rel_path = os.path.join(self.rgb_path, filename_line[0][11:]) 
    #     depth_rel_path, filled_rel_path = None, None
        
    #     if DatasetMode.RGB_ONLY != self.mode:
    #         # e.g., depth/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
    #         depth_rel_path = os.path.join(self.depth_path, filename_line[1])  
    #         if self.has_filled_depth:          
    #             filled_rel_path = filename_line[2]  # e.g., 721.5377
        
    #     return rgb_rel_path, depth_rel_path, filled_rel_path
    
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

    def _get_valid_mask(self, depth: torch.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)  # [1, H, W]

        if self.valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            gt_height, gt_width = eval_mask.shape

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1

            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask


class WeathewrKITTIDepthMixedDataset(WeathewrKITTIDepthDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mix_filenames = self.rgb_files + self.rain_files + self.raingan_files + \
                                self.fog1_files + self.fog2_files + self.snow_files + self.snowgan_files
        print(len(self.mix_filenames), len(self.depth_files))
                  
    def __len__(self):
        return len(self.mix_filenames)
    
    def _get_data_path(self, index):
        rgb_rel_path = self.mix_filenames[index]
        # e.g., rgb/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            # e.g., depth/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
            depth_rel_path = self.depth_files[index]
            if self.has_filled_depth:          
                filled_rel_path = self.filled[index]  # e.g., 721.5377
        return rgb_rel_path, depth_rel_path, filled_rel_path