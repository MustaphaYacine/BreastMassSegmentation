import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)  # y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dsc


def calculate_loss(masks, labels, metrics):
    alpha = 0.8
    dice_loss = DiceLoss()
    bce_loss = nn.BCELoss()

    dice_loss_val = dice_loss(masks, labels)
    bce_loss_val = bce_loss(masks,labels)

    loss = dice_loss_val + alpha * bce_loss_val

    metrics['BCE_loss'] += bce_loss_val.item()
    metrics['Dice_score'] += 1 - dice_loss_val.item()
    metrics['loss'] += loss.item()

    return loss, metrics
