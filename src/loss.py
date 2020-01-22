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
    dice_loss = DiceLoss()
    loss = dice_loss(masks,labels)
    #metrics['IOU'] += 1
    metrics['Dice_score'] += loss.item()
    return loss,metrics