import os
import sys
sys.path.append("../mcts_dl")

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from utils.utils import ReadMapFromMovingAIFile, Map


def get_image_map(gridMap : Map):
    hIm = gridMap.height 
    wIm = gridMap.width
    im = np.zeros((hIm, wIm), dtype=np.uint8)
    for i in range(gridMap.height):
        for j in range(gridMap.width):
            if(gridMap.cells[i][j] == 1):
                im[i][j] = 1
    return np.asarray(im)


class City(Dataset):
    def __init__(self, map_root):
        map_paths = [os.path.join(map_root, path) 
                     for path in os.listdir(map_root) 
                     if ".map" in path]
        
        task_maps = [ReadMapFromMovingAIFile(map_path) 
                     for map_path in map_paths]
        
        self.image_maps = [get_image_map(task_map) 
                           for task_map in task_maps]
        
    def __len__(self):
        return len(self.image_maps)
    
    def __getitem__(self, idx):
        image_map = self.image_maps[idx]
        
        return ToTensor()(image_map)
