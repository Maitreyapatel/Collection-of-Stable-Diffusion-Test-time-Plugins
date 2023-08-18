from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch as th
import torch.nn as nn
from diffusers.models.cross_attention import CrossAttention
from diffusers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class KeyValueTensors(object):
    k: torch.Tensor
    v: torch.Tensor


class StructuredCrossAttention(CrossAttention):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: int = 0,
        struct_attention: bool = False,
    ) -> None:
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)
        self.struct_attention = struct_attention

    def struct_qkv(
        self,
        q: th.Tensor,
        context: Tuple[th.Tensor, KeyValueTensors],
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        assert len(context) == 2 and isinstance(context, tuple)
        uc_context = context[0]
        context_k = context[1].k
        context_v = context[1].v

        if isinstance(context_k, list) and isinstance(context_v, list):
            return self.multi_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        elif isinstance(context_k, th.Tensor) and isinstance(context_v, th.Tensor):
            return self.heterogenous_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        else:
            raise NotImplementedError

    def multi_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> None:
        assert uc_context.size(0) == context_k.size(0) == context_v.size(0)

        # true_bs = uc_context.size(0) * self.heads
        # kv_tensors = self.get_kv(uc_context)

        raise NotImplementedError

    def normal_qkv(
        self,
        q: th.Tensor,
        context: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        batch_size, sequence_length, dim = q.shape

        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._attention(q, k, v, sequence_length, dim)

        return hidden_states

    def heterogenous_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        batch_size, sequence_length, dim = q.shape

        k = self.to_k(th.cat((uc_context, context_k), dim=0))
        v = self.to_v(th.cat((uc_context, context_v), dim=0))

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._attention(q, k, v, sequence_length, dim)

        return hidden_states

    def get_kv(self, context: th.Tensor) -> KeyValueTensors:
        return KeyValueTensors(k=self.to_k(context), v=self.to_v(context))

    def forward(
        self,
        x: th.Tensor,
        context: Optional[Tuple[th.Tensor, KeyValueTensors]] = None,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        q = self.to_q(x)

        if isinstance(context, tuple):
            assert len(context) == 2
            assert isinstance(context[0], th.Tensor)  # unconditioned embedding
            assert isinstance(context[1], KeyValueTensors)  # conditioned embedding

            if self.struct_attention:
                out = self.struct_qkv(q=q, context=context, mask=mask)
            else:
                uc_context = context[0]
                c_full_seq = context[1].k[0].unsqueeze(dim=0)
                out = self.normal_qkv(
                    q=q, context=th.cat((uc_context, c_full_seq), dim=0), mask=mask
                )
        else:
            ctx = context if context is not None else x
            out = self.normal_qkv(q=q, context=ctx, mask=mask)

        return self.to_out(out)


def replace_cross_attention(nn_module: nn.Module, name: str) -> None:
    for attr_str in dir(nn_module):
        target_attr = getattr(nn_module, attr_str)

        if isinstance(target_attr, CrossAttention):
            query_dim = target_attr.to_q.in_features
            assert target_attr.to_k.in_features == target_attr.to_v.in_features
            context_dim = target_attr.to_k.in_features
            heads = target_attr.heads
            dim_head = int(target_attr.scale**-2)
            dropout = target_attr.to_out[-1].p

            sca_kwargs = {
                "query_dim": query_dim,
                "context_dim": context_dim,
                "heads": heads,
                "dim_head": dim_head,
                "dropout": dropout,
            }

            if attr_str == "attn2":
                sca = StructuredCrossAttention(**sca_kwargs, struct_attention=True)
            else:
                sca = StructuredCrossAttention(**sca_kwargs, struct_attention=False)

            original_params = list(target_attr.parameters())
            proposed_params = list(sca.parameters())
            assert len(original_params) == len(proposed_params)

            for p1, p2 in zip(original_params, proposed_params):
                p2.data.copy_(p1.data)

            logger.info(f"Replaced: {name} {attr_str} to {sca}")
            setattr(nn_module, attr_str, sca)

    for name, immediate_child_module in nn_module.named_children():
        replace_cross_attention(nn_module=immediate_child_module, name=name)
