import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class MassSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_dir = sorted([dir for dir in os.listdir(root_dir)])

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, item):
        base_path = os.path.join(self.root_dir, self.list_dir[item])
        (dir_path, dir_names, file_names) = [x for x in os.walk(base_path)][-1]
        img = cv2.imread(dir_path + "/imageReshapedFilterApplied.png")  # we can use just one channel [:,:,0]
        mask = cv2.imread(dir_path + "/maskReshaped.png")
        # change the shape from (H,W,C) to (C,H,W)
        mask = mask.transpose(2, 0, 1)
        mask = np.expand_dims(mask[0,:,:],axis=0).transpose(1,2,0)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask
