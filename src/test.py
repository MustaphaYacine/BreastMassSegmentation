import time
from collections import defaultdict
from loss import calculate_loss


def test(model, test_loader, device='cpu'):
    print("-" * 30, "\nTest:\n")
    start_epoch_time = time.time()
    model.eval()
    metrics = defaultdict(float)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = model(images)
        loss, metrics = calculate_loss(masks, labels, metrics)

    epoch_time = time.time() - start_epoch_time
    print('{:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    print('Best metrics: BCE_loss= {} , Dice_score= {} , || loss = {}'
          .format(metrics['BCE_loss'] / len(test_loader),
                  metrics['Dice_score'] / len(test_loader),
                  metrics['loss'] / len(test_loader)))
