import os
import torch
import random
from .base_depth_dataset import \
    BaseDepthDataset, DepthFileNameMode, DatasetMode


class DrivingStereoDataset(BaseDepthDataset):
    RAW_HEIGHT = 800
    RAW_WIDTH = 1762
    IMGFOLDER = "left-image-full-size"
    DEPTHFOLDER = "depth-map-full-size"
    def __init__(
        self,
        **kwargs):
        super(DrivingStereoDataset, self).__init__(
            min_depth=1e-6,
            max_depth=80,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id_,
            **kwargs)
        self.forename = {
            "rainy": "2018-08-17-09-45-58_2018-08-17-10-", 
            "foggy": "2018-10-25-07-37-26_2018-10-25-"}

        # Load filenames
        self.weather = 'rainy' if 'rainy' in self.filename_ls_path else 'foggy'
    
    def _get_data_path(self, index):
        frame_name = self.forename[self.weather] + self.filenames[index][0] + ".png"
        rgb_rel_path = os.path.join(self.weather, self.IMGFOLDER, frame_name)
        depth_rel_path, filled_rel_path = None, None

        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = os.path.join(self.weather, self.DEPTHFOLDER, frame_name)

        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _load_rgb_data(self, rgb_rel_path, dict_names=("rgb_int", "rgb_norm")):
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_data = {
            dict_names[0]: torch.from_numpy(rgb).int(),
            dict_names[1]: torch.from_numpy(rgb_norm).float(),
        }
        return rgb_data
    
    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        depth_decoded = depth_in / 256.0
        return depth_decoded

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        depth_data = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw.copy()).float().unsqueeze(0)  # [1, H, W]
        depth_data["depth_raw_linear"] = depth_raw_linear.clone()
        
        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled.copy()).float().unsqueeze(0)
            depth_data["depth_filled_linear"] = depth_filled_linear
        else:
            depth_data["depth_filled_linear"] = depth_raw_linear.clone()

        return depth_data

if __name__ == '__main__':
    ds = DrivingStereoDataset(
        filename_ls_path='data_split/drivingstereo_depth/rainy.txt',
        dataset_dir='data/drivingstereo', 
        mode=DatasetMode.EVAL,
        disp_name='DrivingStereoDataset',)
    
    print(len(ds))
    itemd = ds[152]
    print(itemd["index"])
    print(itemd["rgb_relative_path"])
    print(itemd["rgb_int"].shape)
    print(itemd["rgb_norm"].shape)
    print(itemd["depth_filled_linear"].shape)