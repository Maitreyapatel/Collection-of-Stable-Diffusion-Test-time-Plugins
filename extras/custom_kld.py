import numpy as np
import torch
import torch.nn.functional as F
import cv2


def custom_loss_cdist(a, b, p):
    return (
        torch.cdist(torch.tensor(a), torch.tensor(b), p=p).mean()
        + torch.cdist(torch.tensor(a.T), torch.tensor(b.T), p=p).mean()
    ).mean()


def custom_loss(a, b, p):
    loss = 0.0

    a1 = torch.mean(a, dim=0)
    b1 = torch.mean(b, dim=0)
    loss += F.kl_div(a1, b1, reduction="none").mean()

    a2 = torch.mean(a, dim=1)
    b2 = torch.mean(b, dim=1)
    loss += F.kl_div(a2, b2, reduction="none").mean()

    return loss


mask_image1 = np.zeros((512, 512))
mask_image1[10:100, 10:100] = 1.0

mask_image2 = mask_image1

_P = 0

print(
    "lowest:",
    custom_loss(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P),
)


mask_image2 = np.zeros((512, 512))
mask_image2[60:150, 60:150] = 1.0

print(
    "mid 1:",
    custom_loss(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P),
)


mask_image2 = np.zeros((512, 512))
mask_image2[100:200, 100:200] = 1.0

print(
    "mid 2:",
    custom_loss(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P),
)

mask_image2 = np.zeros((512, 512))
mask_image2[350:500, 350:500] = 1.0

print(
    "highest:",
    custom_loss(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P),
)
