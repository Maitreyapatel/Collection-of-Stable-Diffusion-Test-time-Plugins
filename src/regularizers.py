import math

import torch
import torch.nn.functional as F

from utils.ptp_utils import *
from utils.gaussian_smoothing import GaussianSmoothing




def get_layout_guidance_loss(controller):
    def _compute_loss(b_bbox, b_object_positions, attention_res = 16, smooth_attentions = False, kernel_size = 3, sigma = 0.5, normalize_eot = False, device="cuda") -> torch.Tensor:
        attention_maps = aggregate_attention_batched(
            controller,
            res=16,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0
        )
        loss = 0.0
        for idx in range(len(attention_maps)):
            attention_for_text = attention_maps[idx]
            bbox = b_bbox[idx]
            object_positions = b_object_positions[idx]
            H, W, n = attention_for_text.shape 
            for obj_idx in range(len(bbox)):
                obj_loss = 0.0
                mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                for obj_box in bbox[obj_idx]: ## TODO: why there is this loop? there should be only one bbox per object token. But this might be useful for Spatial Attend & Excite
                    x_min, y_min, x_max, y_max = round(obj_box[0] * W / 64), \
                        round(obj_box[1] * H / 64), round(obj_box[2] * W / 64), round(obj_box[3] * H / 64) ###### -> scaling with 64 because the attention map is 16x16 and the mask is 64x64
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
            
            loss += loss / len(bbox)

        return loss / len(attention_maps)
    
    return _compute_loss



def get_attend_and_excite_loss(controller):
    ## TODO: This loss do not support greater than 1 batch size
    def _compute_loss(bbox, object_positions, attention_res = 16, smooth_attentions = False, kernel_size = 3, sigma = 0.5, normalize_eot = False, device="cuda") -> torch.Tensor:
        attention_maps = aggregate_attention(
            attention_store=controller,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        
        attention_for_text = torch.stack(attention_maps)#[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token 
        # Not performing shifting as 
        indices_to_alter = [index for index in object_positions]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())

        max_attention_per_index = max_indices_list
        

        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        return loss
    
    return _compute_loss



def get_attention_refocus_loss(controller, w1=1.0):
    ## TODO: This loss do not support greater than 1 batch size
    def _compute_loss(bbox, object_positions, attention_res = 16, smooth_attentions = False, kernel_size = 3, sigma = 0.5, normalize_eot = False, device="cuda") -> torch.Tensor:
        self_attention_maps = []
        attention_res = [attention_res]
        for res in attention_res:
            maps = only_SAR(
                attention_store=controller,
                aggregation_method="aggregate_attention",
                res=res,
                from_where=["down", "mid", "up"],
                select=0)
            if maps is not None:
                self_attention_maps.append(maps)

        cross_attention_maps = []
        for res in attention_res:
            maps = only_CAR(
                attention_store=controller,
                aggregation_method="aggregate_attention",
                res=res, 
                from_where=["down", "mid", "up"],
                select=0)
            if maps is not None:
                cross_attention_maps.append(maps)
        

        def loss_one_att_outside(attn_map,bboxes):
            loss = 0
            object_number = len(bboxes)
            b, i, j = attn_map.shape
            H = W = 4#int(math.sqrt(i))
            
            for obj_idx in range(object_number):
                
                for idx, obj_box in enumerate(bboxes[obj_idx]):
                    mask = torch.zeros(size=(i, j)).cuda() if torch.cuda.is_available() else torch.zeros(size=(i, j))
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                        int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                    mask[y_min: y_max, x_min: x_max] = 1.
                    mask_out = 1. - mask

                    # index = (mask == 1.).nonzero(as_tuple=False)
                    # index_in_key = index[:,0]*i + index[:, 1]
                    # att_box = torch.zeros_like(attn_map)
                    # att_box[:,index_in_key,:] = attn_map[:,index_in_key,:]

                    # att_box = att_box.sum(axis=1) / index_in_key.shape[0]
                    # att_box = att_box.reshape(-1, i, j)
                    activation_value = (attn_map* mask)#.reshape(b, -1).sum(dim=-1)
                    loss += torch.mean(activation_value)
                    
            return loss / object_number

        def loss_one_att_inside(attn_map,bboxes,object_positions, smooth_attentions, kernel_size, sigma):
            
            obj_number = len(bboxes)
            total_loss = 0
            attn_text = attn_map[:, :, 1:-1]
            attn_text *= 100                                          ## scaling and activation added
            attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
            H = W = attn_map.shape[1]

            for obj_idx in range(len(bbox)):
                for obj_position in object_positions[obj_idx]:
                    true_obj_position = obj_position - 1
                    att_map_obj = attn_text[:, :, true_obj_position]
                    if smooth_attentions:
                        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                        input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                        att_map_obj = smoothing(input).squeeze(0).squeeze(0)
                    other_att_map_obj = att_map_obj.clone()
                    att_copy = att_map_obj.clone()

                    for obj_box in bboxes[obj_idx]:
                        x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                        int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                    
                    
                        if att_map_obj[y_min: y_max, x_min: x_max].numel() == 0: 
                            max_inside=1.
                            
                        else:
                            max_inside = att_map_obj[y_min: y_max, x_min: x_max].max()
                        total_loss += 1. - max_inside
                        
                        # find max outside the box, find in the other boxes
                        
                        att_copy[y_min: y_max, x_min: x_max] = 0.
                        other_att_map_obj[y_min: y_max, x_min: x_max] = 0.
    
                    for obj_outside in range(obj_number):
                        if obj_outside != obj_idx:
                            for obj_out_box in bboxes[obj_outside]:
                                x_min_out, y_min_out, x_max_out, y_max_out = int(obj_out_box[0] * W), \
                                    int(obj_out_box[1] * H), int(obj_out_box[2] * W), int(obj_out_box[3] * H)
                                
                                # att_copy[y_min: y_max, x_min: x_max] = 0.
                                if other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].numel() == 0: 
                                    max_outside_one= 0
                                else:
                                    max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].max()
                                # max_outside = max(max_outside,max_outside_one )
                                att_copy[y_min_out: y_max_out, x_min_out: x_max_out] = 0.
                                total_loss += max_outside_one
                    max_background = att_copy.max()
                    total_loss += len(bboxes[obj_idx]) *max_background /2.

            return total_loss


        def caculate_loss_self_att(self_attentions, bboxes):
            cnt = 0
            total_loss = 0
            for position in self_attentions:
                for attn in position:
                    ## reshape attn from shape (b, h, w, n) to (b, h*w, n)
                    attn = attn.reshape(attn.shape[0], -1, attn.shape[-1])
                    total_loss += loss_one_att_outside(attn, bboxes)
                    cnt += 1

            return total_loss / cnt

        def caculate_loss_cross_att(cross_attentions, bboxes, object_positions, smooth_attentions, kernel_size, sigma):
            obj_num = len(bboxes)
            total_loss = 0
            for position in cross_attentions:
                for attn in position:
                    total_loss += loss_one_att_inside(attn, bboxes, 
                            object_positions, smooth_attentions, kernel_size, sigma)

            return total_loss / obj_num



        self_loss = caculate_loss_self_att(self_attention_maps, bbox)
        cross_loss = caculate_loss_cross_att(cross_attention_maps, bbox, object_positions, smooth_attentions, kernel_size, sigma)

        return self_loss*w1 + cross_loss
    
    return _compute_loss