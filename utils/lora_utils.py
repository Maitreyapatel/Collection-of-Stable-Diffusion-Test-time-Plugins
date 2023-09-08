import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention


def register_attention_control_unet_lora(unet, controller, lora_rank=8):
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    cross_att_count = 0
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
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

        # if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
        #     lora_attn_processor_class = LoRAAttnAddedKVProcessor
        # else:
        #     lora_attn_processor_class = (
        #         LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        #     )
        lora_attn_processor_class = LoRAAttnProcessor
        cross_att_count += 1

        module = lora_attn_processor_class(
            attnstore=controller,
            place_in_unet=place_in_unet,
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=lora_rank,
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)
    controller.num_att_layers = cross_att_count

    return unet_lora_parameters


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        network_alpha=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAAttnAddedKVProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
    encoder.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.

    """

    def __init__(
        self,
        attnstore,
        place_in_unet,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
    ):
        super().__init__()

        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.add_k_proj_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.add_v_proj_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_k_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(
            hidden_size, hidden_size, rank, network_alpha
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        is_cross = encoder_hidden_states is not None

        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(
            encoder_hidden_states
        ) + scale * self.add_k_proj_lora(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(
            encoder_hidden_states
        ) + scale * self.add_v_proj_lora(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states)
            value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(
        self,
        attnstore,
        place_in_unet,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
    ):
        super().__init__()

        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(
            hidden_size, hidden_size, rank, network_alpha
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        temb=None,
    ):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRAAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(
        self,
        attnstore,
        place_in_unet,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(
            hidden_size, hidden_size, rank, network_alpha
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        is_cross = encoder_hidden_states is not None

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )

        with torch.no_grad():
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
