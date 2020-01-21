from .model import UNet
from .dataset import MassSegmentationDataset
from .train import train
from .test import test
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def combined_transform(img,mask):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img),transform(mask)
def dataloaders(train_root_dir, validation_root_dir, transform, batch_size=4):
    train_loader = DataLoader(
        MassSegmentationDataset(train_root_dir, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        MassSegmentationDataset(validation_root_dir),  # no trransform for validation
        batch_size=batch_size,
        num_workers=0
    )
    return {'train': train_loader,
            'valid': valid_loader}

train_root_dir = '/**'
validation_root_dir = '/**'
test_root_dir = '/**'
batch_size=4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loaders
loaders = dataloaders(train_root_dir,validation_root_dir,combined_transform,batch_size)

model = UNet(in_channels=3, out_channels=1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.3)
scheduler = None # not yet

model = train(model, optimizer, scheduler, loaders, nb_epochs=20, device='cpu', path_weights='./')

test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
test_loader = DataLoader(
        MassSegmentationDataset(test_root_dir, test_transform),
        batch_size=batch_size,
        num_workers=0
    )

test(model, test_loader, device)