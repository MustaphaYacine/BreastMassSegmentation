from model import UNet
from dataset import MassSegmentationDataset
from train import train
from test import test
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np


def combined_transform(img, mask):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_mask = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform_img(img), transform_mask(mask)


def dataloaders(train_root_dir, transform, batch_size=4, valid_size=0.2):
    train_data = MassSegmentationDataset(train_root_dir, transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0
    )
    valid_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0
    )
    return {'train': train_loader,
            'validation': valid_loader}


def main():
    train_root_dir = '/content/drive/My Drive/DDSM/train/CBIS-DDSM'
    test_root_dir = '/content/drive/My Drive/DDSM/test/CBIS-DDSM'
    batch_size = 2
    valid_size = 0.2
    nb_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loaders
    loaders = dataloaders(train_root_dir, combined_transform, batch_size, valid_size)

    model = UNet(in_channels=3, out_channels=1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.3)
    scheduler = None  # not yet

    model = train(model, optimizer, scheduler, loaders, nb_epochs, device, path_weights='./')
    # from torchsummary import summary
    #
    # summary(model, input_size=(3, 224, 224))

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    test_loader = DataLoader(
        MassSegmentationDataset(test_root_dir, combined_transform),
        batch_size=batch_size,
        num_workers=0
    )

    test(model, test_loader, device)


# if __name__ == '__main__':
main()
