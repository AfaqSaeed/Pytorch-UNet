import torch
import torch.nn.functional as F
from tqdm import tqdm

# from utils.dice_score import multiclass_dice_coeff, dice_coeff

criterion = torch.nn.L1Loss()
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype= torch.uint8)

            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            # compute the Dice score, ignoring background
            loss += criterion(mask_pred,mask_true)

    net.train()
    return loss / max(num_val_batches, 1)
