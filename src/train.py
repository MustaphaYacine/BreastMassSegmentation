##
import time
from collections import defaultdict
##
import torch
from loss import calculate_loss


def train(model, optimizer, scheduler, dataloaders, nb_epochs=20, device='cpu', path_weights='./weights'):
    best_score = {'loss': 1e10}
    for epoch in range(nb_epochs):
        print("Epoch {}/{}".format(epoch + 1, nb_epochs))
        start_epoch_time = time.time()
        for phase in ['train', 'validation']:
            print('----- {} -----'.format(phase))
            if phase == 'train':
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_size = 0
            i = 0
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                i += 1
                with torch.set_grad_enabled(phase == 'train'):
                    masks = model(images)
                    loss, metrics = calculate_loss(masks, labels, metrics)
                    if i % 150 == 0:
                        print('i == {}, loss == {}'.format(i, loss.item()))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_size += images.size(0)
            epoch_loss = metrics['loss'] / len(dataloaders[phase])  # // epoch_size
            if phase == 'validation':
                if scheduler:
                    scheduler.step(epoch_loss)
            if phase == 'validation' and epoch_loss < best_score['loss']:
                print('best model updated, loss *** changed from {}, to {}'
                      .format(best_score['loss'], epoch_loss ))
                best_score = {'loss': epoch_loss, 'BCE_loss': metrics['BCE_loss'] / len(dataloaders[phase]),
                              'Dice_score': metrics['Dice_score'] / len(dataloaders[phase])}
                print('metrics: BCE_loss= {} , Dice_score= {}'
                      .format(best_score['BCE_loss'],best_score['Dice_score']))

                torch.save(model.state_dict(), path_weights + '.pt')

        epoch_time = time.time() - start_epoch_time
        print('{:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    print('Best metrics: BCE_loss= {} , Dice_score= {} , || loss = {}'
          .format(best_score['BCE_loss'],
                  best_score['Dice_score'],
                  best_score['loss']))  # Dice score

    model.load_state_dict(torch.load(path_weights + '.pt'))
    return model
