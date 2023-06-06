import os
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MayoClinicDataset(Dataset):
    def __init__(self, csv_file:Union[str, pd.DataFrame], root_dir:str, transform:transforms.Compose=None) -> None:
        super().__init__()
        self.tabular_data = pd.read_csv(csv_file) if isinstance(csv_file, str) else csv_file
        self.root_dir = root_dir
        self.transform = transform
        
        self.train = 'label' in self.tabular_data.columns
        self.classes, self.class_to_idx = self._find_classes(self.tabular_data) if self.train else (['CE', 'LAA'], {'CE':0, 'LAA':1})
        
    def __len__(self) -> int:
        return len(self.tabular_data)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.tabular_data['image_id'].iloc[index])
        image = Image.open(f'{img_path}.png')
        
        label = self.class_to_idx[self.tabular_data['label'].iloc[index]] if self.train else -1
        
        if self.transform:
            image = self.transform(image)
            
        return (image, label)
    
    def _find_classes(self, tabular_data:pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
        classes = list(sorted(tabular_data['label'].unique()))
        class_to_idx = {classes[i]:i for i in range(len(classes))}
        
        return classes, class_to_idx
    