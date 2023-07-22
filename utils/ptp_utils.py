import abc
import sys
import importlib
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.cross_attention import CrossAttention

def Pharse2idx(prompt, phrases):
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

def register_attention_control_unet(unet, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # print('item.shape', item.shape)
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    if out:
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return [out]
    return None

def aggregate_attention_SAR_CAR(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    
    attention_maps = attention_store.get_average_attention()

    out = []
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # print('item.shape', item.shape)
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    if out and is_cross:
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return [out]
    elif out:
        return out
    return None

#get all attentions 'cross and self' using 'from_where'
def all_attention(attention_store: AttentionStore,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """up and mid cross attentions"""
    out = []
    attention_maps = attention_store.get_average_attention()
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            resolution = torch.sqrt(torch.tensor(item.shape[1])).int()
            cross_map = item.reshape(1, -1, resolution, resolution, item.shape[-1])[select]
            for i in range(cross_map.shape[0]):
                out.append(cross_map[i])
    return out

def aggregate_layer_attention(attention_store: AttentionStore,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """up and mid cross attentions, aggregate accross layers"""
    out = []
    layer_agg = []
    counter = 0
    attention_maps = attention_store.get_average_attention()
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            res = torch.sqrt(torch.tensor(item.shape[1])).int()
            if counter != 0:
                if res!=current_res:
                    layer_agg = torch.cat(layer_agg, dim=0)
                    layer_agg = layer_agg.sum(0) / layer_agg.shape[0]
                    out.append(layer_agg)
                    current_res = res
                    layer_agg = []
            current_res = res
            cross_map = item.reshape(1, -1, res, res, item.shape[-1])[select]
            layer_agg.append(cross_map)
            counter+=1
    layer_agg = torch.cat(layer_agg, dim=0)
    layer_agg = layer_agg.sum(0) / layer_agg.shape[0]
    out.append(layer_agg)
    return out

def only_CAR(attention_store: AttentionStore,
                        aggregation_method: str,
                        res: int,
                        from_where: List[str],
                        select: int) -> torch.Tensor:
    """Cross Attention Refocusing"""
    if aggregation_method == 'aggregate_attention':
        out = aggregate_attention_SAR_CAR(attention_store, res, from_where, True, select)
    elif aggregation_method == 'aggregate_layer_attention':
        out = aggregate_layer_attention(attention_store, from_where, True, select)
    elif aggregation_method == 'all_attention':
        out = all_attention(attention_store, from_where, True, select)
    else:
        raise NotImplementedError
    return out

def only_SAR(attention_store: AttentionStore,
                        aggregation_method: str,
                        res: int,
                        from_where: List[str],
                        select: int) -> torch.Tensor:
    """Self Attention Refocusing"""
    if aggregation_method == 'aggregate_attention':
        out = aggregate_attention_SAR_CAR(attention_store, res, from_where, False, select)
    elif aggregation_method == 'aggregate_layer_attention':
        out = aggregate_layer_attention(attention_store, from_where, False, select)
    elif aggregation_method == 'all_attention':
        out = all_attention(attention_store, from_where, False, select)
    else:
        raise NotImplementedError
    return out

def CAR_SAR(attention_store: AttentionStore,
                        aggregation_method: str,
                        res: 16,
                        from_where: List[str],
                        select: int) -> torch.Tensor:
    """Cross Attention Refocusing + Self Attention Refocusing"""
    if aggregation_method == 'aggregate_attention':
        out = aggregate_attention_SAR_CAR(attention_store, res, from_where, True, select)
        out += aggregate_attention_SAR_CAR(attention_store, res, from_where, False, select)
    elif aggregation_method == 'aggregate_layer_attention':
        out = aggregate_layer_attention(attention_store, from_where, True, select)
        out += aggregate_layer_attention(attention_store, from_where, False, select)
    elif aggregation_method == 'all_attention':
        out = all_attention(attention_store, from_where, True, select)
        out += all_attention(attention_store, from_where, False, select)
    else:
        raise NotImplementedError
    return out
