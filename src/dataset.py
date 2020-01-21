import os
import cv2
from torch.utils.data import Dataset


class MassSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_dir = sorted([dir for dir in os.listdir(root_dir)])

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, item):
        path = os.path.join(self.root_dir, self.list_dir[item])
        img = cv2.imread(os.listdir(path)[0])  # we can use just one channel [:,:,0]
        mask = cv2.imread(os.listdir(path)[-1])[:, :, 0]
        # change the shape from (H,W,C) to (C,H,W)
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask
