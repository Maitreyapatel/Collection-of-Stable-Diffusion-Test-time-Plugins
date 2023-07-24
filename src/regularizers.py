import torch
import torch.nn.functional as F

from utils.ptp_utils import *
from utils.gaussian_smoothing import GaussianSmoothing




def get_layout_guidance_loss(controller):
    ## TODO: This loss do not support greater than 1 batch size
    def _compute_loss(bbox, object_positions, attention_res = 16, smooth_attentions = False, kernel_size = 3, sigma = 0.5, normalize_eot = False, device="cuda") -> torch.Tensor:
        attention_maps = aggregate_attention(
            controller,
            res=16,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0
        )
        

        loss = 0.0
        for attention_for_text in attention_maps:
            H = W = attention_for_text.shape[1]
            for obj_idx in range(len(bbox)):
                obj_loss = 0.0
                mask = torch.zeros(size=(H, W)).to(device)
                for obj_box in bbox[obj_idx]: 
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                        int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                    mask[y_min: y_max, x_min: x_max] = 1

                for obj_position in object_positions[obj_idx]:
                    image = attention_for_text[:, :, obj_position]
                    if smooth_attentions:
                        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                        input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                        image = smoothing(input).squeeze(0).squeeze(0)
                    activation_value = (image * mask).reshape(image.shape[0], -1).sum(dim=-1)/image.reshape(image.shape[0], -1).sum(dim=-1)
                    obj_loss += torch.mean((1 - activation_value) ** 2)
                loss += obj_loss / len(object_positions[obj_idx])

        loss = loss / (len(bbox) * len(attention_maps))
        return loss
    
    return _compute_loss