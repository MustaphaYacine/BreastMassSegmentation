import argparse
import json
import os
##
import time
from collections import defaultdict
##
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loss import calculate_loss



def train(model, optimizer, scheduler, dataloaders, nb_epochs=20, device='cpu', path_weights='./'):
    best_score = 1e10
    for epoch in range(nb_epochs):
        print("Epoch {}/{}".format(epoch + 1, nb_epochs))
        start_epoch_time = time.time()
        for phase in ['train', 'validation']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_size = 0
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    masks = model(images)
                    loss, metrics = calculate_loss(masks, labels, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_size += images.size(0)
            epoch_loss = metrics['Dice_score']/len(dataloaders[phase]) # // epoch_size
            if phase == 'validation' and epoch_loss < best_score:
                print('best model updated, metric *** changed from {}, to {}'.format(best_score, metrics['Dice_score']))
                best_score = epoch_loss
                torch.save(model.state_dict(), path_weights + '.pt')
        epoch_time = time.time() - start_epoch_time
        print('{:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    print('Best metric val : {:4f}'.format(1-best_score)) # Dice score

    model.load_state_dict(torch.load(path_weights + '.pt'))
    return model




