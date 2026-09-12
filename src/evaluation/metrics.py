import torch

def dice_coeff(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient.
    pred and target are both tensors of shape (N, C, H, W)
    with values in [0, 1].
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_coeff(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - dice_coeff(pred, target)
