import torch
import torch.nn.functional as F
from tqdm import tqdm

from loam.utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            #image, mask_true = batch['image'], batch['mask']
            #image, mask_true, mask_true_sup = batch['image'], batch['mask'], batch['mask_sup']
            image, mask_true, auxiliary_info_1, auxiliary_info_2 = batch['image'], batch['mask'], batch['auxiliary_1'], batch['auxiliary_2']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            auxiliary_info_1 = auxiliary_info_1.to(device=device, dtype=torch.float32)
            auxiliary_info_2 = auxiliary_info_2.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            #mask_true_sup = mask_true_sup.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image, auxiliary_info_1, auxiliary_info_2)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            
            '''
            # predict the mask
            mask_pred = net(image)[0]
            mask_pred_sup = net(image)[1]

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_score += dice_coeff(mask_pred_sup, mask_true_sup, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                mask_true_sup = F.one_hot(mask_true_sup, net.n_classes).permute(0, 3, 1, 2).float()
                mask_true_sup = F.one_hot(mask_true_sup.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                dice_score += multiclass_dice_coeff(mask_pred_sup[:, 1:], mask_true_sup[:, 1:], reduce_batch_first=False)
            '''

    net.train()
    return dice_score / max(num_val_batches, 1)