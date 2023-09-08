import numpy as np
import torch
import cv2

mask_image1 = np.zeros((512, 512))
mask_image1[10:100, 10:100] = 1.0

mask_image2 = mask_image1

_P = float("inf")

print(
    "lowest:",
    torch.cdist(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P).mean(),
)


mask_image2 = np.zeros((512, 512))
mask_image2[10:150, 10:150] = 1.0

print(
    "mid 1:",
    torch.cdist(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P).mean(),
)


mask_image2 = np.zeros((512, 512))
mask_image2[100:200, 100:200] = 1.0

print(
    "mid 2:",
    torch.cdist(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P).mean(),
)

mask_image2 = np.zeros((512, 512))
mask_image2[400:500, 400:500] = 1.0

print(
    "highest:",
    torch.cdist(torch.tensor(mask_image1), torch.tensor(mask_image2), p=_P).mean(),
)
