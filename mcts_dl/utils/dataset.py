import os
import numpy as np
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset
from .utils import ReadMapFromMovingAIFile, Map


def grid_map2image(gridMap: Map):
    hIm = gridMap.height
    wIm = gridMap.width
    im = np.zeros((hIm, wIm), dtype=np.uint8)
    for i in range(hIm):
        for j in range(wIm):
            if gridMap.cells[i][j] == 1:
                im[i][j] = 1
    return np.asarray(im)


class City(Dataset):
    def __init__(self, map_root):
        self.map_paths = [path for path in os.listdir(map_root) if "_256.map" in path]

        task_maps = [ReadMapFromMovingAIFile(os.path.join(map_root, map_path))
                     for map_path in self.map_paths]

        self.image_maps = [grid_map2image(task_map) for task_map in task_maps]

    def __len__(self):
        return len(self.image_maps)

    def __getitem__(self, idx):
        sample = dict()
        name = self.map_paths[idx]
        image_map = self.image_maps[idx]
        image_map = torch.from_numpy(image_map).float()
        sample['name'] = name
        sample['image'] = torch.unsqueeze(image_map, 0)

        return sample