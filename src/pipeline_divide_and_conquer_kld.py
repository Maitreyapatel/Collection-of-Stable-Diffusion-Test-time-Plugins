import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import (
    CLIPFeatureExtractor,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import (
    AttentionStore,
    aggregate_attention,
    all_attention,
    aggregate_layer_attention,
)
from utils.ptp_retrieval_utils import AttentionRetrievalStore
from utils.attention_utils import *

import copy

logger = logging.get_logger(__name__)

import os
import cv2
import math
from PIL import Image
from utils import ptp_utils


def show_cross_attention(
    prompt: str,
    attention_maps,
    tokenizer,
    indices_to_alter: List[int],
    res: int = 16,
    orig_image=None,
):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i].detach().cpu().mean(dim=0)
        if i in indices_to_alter:
            image = show_image_relevance(image, relevnace_res=256)
            image = np.uint8(255 * image)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(np.tile(np.expand_dims(image, axis=-1), (1, 1, 3)))

    return ptp_utils.view_images(np.stack(images, axis=0), display_image=False)


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def show_image_relevance(image_relevance, relevnace_res=256):
    image_relevance = image_relevance.reshape(
        1,
        1,
        int(math.sqrt(image_relevance.shape[-1])),
        int(math.sqrt(image_relevance.shape[-1])),
    )
    image_relevance = (
        image_relevance.cuda()
    )  # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=relevnace_res, mode="bilinear"
    )
    image_relevance = image_relevance.cpu()  # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    image_relevance = image_relevance.reshape(relevnace_res, relevnace_res)
    return image_relevance


class DivideAndConquerPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        self.retrieval_unet_sub1 = copy.deepcopy(unet).cuda()
        self.retrieval_unet_sub2 = copy.deepcopy(unet).cuda()
        self.retrieval_scheduler_sub1 = copy.deepcopy(scheduler)
        self.retrieval_scheduler_sub2 = copy.deepcopy(scheduler)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def get_backward_guidance_loss(
        self,
        attention_maps: List[torch.Tensor],
        bbox: Union[int, List[float]],
        object_positions,
        attention_res: int = 16,
        smooth_attentions: bool = False,
        kernel_size: int = 3,
        sigma: float = 0.5,
        normalize_eot: bool = False,
    ) -> torch.Tensor:
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1

        loss = 0.0
        for attention_for_text in attention_maps:
            H = W = attention_for_text.shape[1]
            for obj_idx in range(len(bbox)):
                obj_loss = 0.0
                mask = (
                    torch.zeros(size=(H, W)).cuda()
                    if torch.cuda.is_available()
                    else torch.zeros(size=(H, W))
                )
                for obj_box in bbox[
                    obj_idx
                ]:  ## TODO: why there is this loop? there should be only one bbox per object token. But this might be useful for Spatial Attend & Excite
                    x_min, y_min, x_max, y_max = (
                        int(obj_box[0] * W),
                        int(obj_box[1] * H),
                        int(obj_box[2] * W),
                        int(obj_box[3] * H),
                    )
                    mask[y_min:y_max, x_min:x_max] = 1

                for obj_position in object_positions[obj_idx]:
                    image = attention_for_text[:, :, obj_position]
                    if smooth_attentions:
                        smoothing = GaussianSmoothing(
                            channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
                        ).cuda()
                        input = F.pad(
                            image.unsqueeze(0).unsqueeze(0),
                            (1, 1, 1, 1),
                            mode="reflect",
                        )
                        image = smoothing(input).squeeze(0).squeeze(0)
                    activation_value = (image * mask).reshape(image.shape[0], -1).sum(
                        dim=-1
                    ) / image.reshape(image.shape[0], -1).sum(dim=-1)
                    obj_loss += torch.mean((1 - activation_value) ** 2)
                loss += obj_loss / len(object_positions[obj_idx])

        loss = loss / (len(bbox) * len(attention_maps))
        return loss

    def _get_attention_maps(
        self,
        attention_aggregation_method,
        attention_store: AttentionStore,
        attention_res: int = 16,
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        if attention_aggregation_method == "aggregate_attention":
            attention_maps = aggregate_attention(
                attention_store=attention_store,
                res=attention_res,  ## TODO: We might need to keep original resolutions
                from_where=("up", "mid", "down"),
                is_cross=True,
                select=0,
            )
        elif attention_aggregation_method == "all_attention":
            attention_maps = all_attention(
                attention_store=attention_store,
                from_where=("up", "mid", "down"),
                is_cross=True,
                select=0,
            )
        elif attention_aggregation_method == "aggregate_layer_attention":
            attention_maps = aggregate_layer_attention(
                attention_store=attention_store,
                from_where=("up", "mid", "down"),
                is_cross=True,
                select=0,
            )
        else:
            raise ValueError("Invalid attention aggregation method")
        return attention_maps

    @staticmethod
    def _compute_loss_jsd(
        attn1: List[torch.Tensor],
        attn2: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""

        def bind_loss(attn1, attn2):
            loss = 0.0
            total_m = 0.5 * (attn1 + attn2)
            loss += F.kl_div(attn1.log(), total_m, reduction="mean")
            loss += F.kl_div(attn2.log(), total_m, reduction="mean")
            return loss

        return bind_loss(attn1, attn2)

    @staticmethod
    def _compute_loss(
        attn1: List[torch.Tensor],
        attn2: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""

        loss = 0.0
        loss += F.kl_div(attn1, attn2, reduction="none").mean()
        # loss += F.kl_div(attn2, attn1, reduction="none").mean()
        return loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def tokenize(self, prompt: Union[str, List[str]]):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input

    def _align_sequence(
        self,
        full_seq: torch.Tensor,
        seq: torch.Tensor,
        span: int,
        eos_loc: int,
        dim: int = 1,
        zero_out: bool = False,
        replace_pad: bool = False,
    ) -> torch.Tensor:
        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        # shape: (77, 768) -> (768, 77)
        full_seq = full_seq.transpose(0, dim)

        seg_length = 1

        full_seq[span] = seq[span]
        if zero_out:
            full_seq[1:span] = 0
            full_seq[span:eos_loc] = 0

        # TODO: there might be bug on dimentions checkout the issue: https://github.com/shunk031/training-free-structured-diffusion-guidance/issues/11
        if replace_pad:
            pad_length = len(full_seq) - eos_loc
            full_seq[eos_loc:] = seq[1 + seg_length : 1 + seg_length + pad_length]

        # shape: (768, 77) -> (77, 768)
        return full_seq.transpose(0, dim)

    def align_seq(self, nps: List[str], spans: List[int]) -> KeyValueTensors:
        input_ids = self.tokenize(nps).input_ids
        nps_length = [len(ids) - 2 for ids in input_ids]
        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        k_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        # shape: (num_nps, model_max_length, hidden_dim)
        v_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        return KeyValueTensors(k=k_c, v=v_c)

    def apply_text_encoder(
        self,
        prompt: str,
        nps: List[str],
        spans: Optional[List[int]] = None,
        struct_attention="none",
    ) -> Union[torch.Tensor, KeyValueTensors]:
        # if struct_attention == "extend_str":
        #     return self.extend_str(nps=nps)

        # elif struct_attention == "extend_seq":
        #     return self.extend_seq(nps=nps)

        if struct_attention == "align_seq" and spans is not None:
            return self.align_seq(nps=nps, spans=spans)

        elif struct_attention == "none":
            text_input = self.tokenize(prompt)
            return self.text_encoder(text_input.input_ids.to(self.device))[0]

        else:
            raise ValueError(f"Invalid type of struct attention: {struct_attention}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        attention_store_sub1: AttentionStore,
        attention_store_sub2: AttentionStore,
        retrieval_attention_store: AttentionRetrievalStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_iter_to_alter: Optional[int] = 25,
        run_standard_sd: bool = False,
        thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1.0, 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        sd_2_1: bool = False,
        cfg=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.attention_store_sub1 = attention_store_sub1
        self.attention_store_sub2 = attention_store_sub2

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        cond_embeddings = self.apply_text_encoder(
            struct_attention="align_seq",
            prompt=prompt,
            nps=[prompt, cfg.token_a, cfg.token_b],
            spans=[None, cfg.token_indices[0][0][0][0], cfg.token_indices[1][0][0][0]],
        )

        # text_inputs_main, prompt_embeds_main = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        # )

        # 3.a Encode input prompt_a
        text_inputs_a, prompt_embeds_a = self._encode_prompt(
            cfg.prompt_a,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # 3.b Encode input prompt_b
        text_inputs_b, prompt_embeds_b = self._encode_prompt(
            cfg.prompt_b,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.retrieval_scheduler_sub1.set_timesteps(num_inference_steps, device=device)
        self.retrieval_scheduler_sub2.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            cond_embeddings.k.dtype,
            device,
            generator,
            latents,
        )

        latents_sub1 = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            cond_embeddings.k.dtype,
            device,
            generator,
            None,
        )
        latents_sub2 = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            cond_embeddings.k.dtype,
            device,
            generator,
            None,
        )

        # latents_sub1 = copy.deepcopy(latents)
        # latents_sub2 = copy.deepcopy(latents)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    # process the latents for prompt_a
                    latents_sub1 = latents_sub1.clone().detach().requires_grad_(True)
                    # process the latents for prompt_b
                    latents_sub2 = latents_sub2.clone().detach().requires_grad_(True)

                    # predict the noise residual for prompt_a --> store the attentions in attention_store
                    noise_pred_sub1 = self.retrieval_unet_sub1(
                        latents_sub1,
                        t,
                        encoder_hidden_states=prompt_embeds_a[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    self.retrieval_unet_sub1.zero_grad()

                    # predict the noise residual for prompt_b --> store the attentions in attention_store
                    noise_pred_sub2 = self.retrieval_unet_sub2(
                        latents_sub2,
                        t,
                        encoder_hidden_states=prompt_embeds_b[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    self.retrieval_unet_sub2.zero_grad()

                    attention_maps_sub1 = aggregate_attention(
                        attention_store=self.attention_store_sub1,
                        res=attention_res,
                        from_where=("up", "down", "mid"),
                        is_cross=True,
                        select=0,
                    )[0][:, :, cfg.token_indices[0][1][0][0]]
                    attention_maps_sub2 = aggregate_attention(
                        attention_store=self.attention_store_sub2,
                        res=attention_res,
                        from_where=("up", "down", "mid"),
                        is_cross=True,
                        select=0,
                    )[0][:, :, cfg.token_indices[1][1][0][0]]

                    loss = 10 * self._compute_loss(
                        attention_maps_sub1, attention_maps_sub2
                    )

                    latents_sub1 = self._update_latent(
                        latents=latents_sub1,
                        loss=loss,
                        step_size=scale_factor,
                    )
                    latents_sub2 = self._update_latent(
                        latents=latents_sub2,
                        loss=loss,
                        step_size=scale_factor,
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # process the latents for prompt_a
                latent_model_input_sub1 = (
                    torch.cat([latents_sub1] * 2)
                    if do_classifier_free_guidance
                    else latents_sub1
                )
                latent_model_input_sub1 = (
                    self.retrieval_scheduler_sub1.scale_model_input(
                        latent_model_input_sub1, t
                    )
                )

                # process the latents for prompt_b
                latent_model_input_sub2 = (
                    torch.cat([latents_sub2] * 2)
                    if do_classifier_free_guidance
                    else latents_sub2
                )
                latent_model_input_sub2 = (
                    self.retrieval_scheduler_sub2.scale_model_input(
                        latent_model_input_sub2, t
                    )
                )

                if do_classifier_free_guidance:
                    uncond_input = self.tokenize([""] * batch_size)
                    uncond_embeddings = self.text_encoder(
                        uncond_input.input_ids.to(self.device)
                    )[0]

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    struct_attention = "align_seq"
                    if struct_attention == "align_seq":
                        # shape (uncond_embeddings): (1, model_max_length, hidden_dim)
                        # shape (cond_embeddings):
                        # KeyValueTensors.v (num_nps, model_max_length, hidden_dim)
                        # KeyValueTensors.k (num_nps, model_max_length, hidden_dim)
                        prompt_embeds_main = (uncond_embeddings, cond_embeddings)
                    else:
                        # shape (uncond_embeddings): (1, model_max_length, hidden_dim)
                        # shape (cond_embeddings): (num_nps, model_max_length, hidden_dim)
                        # shape: (1 + num_nps, model_max_length, hidden_dim)
                        prompt_embeds_main = torch.cat(
                            [uncond_embeddings, cond_embeddings]
                        )
                else:
                    prompt_embeds_main = cond_embeddings

                # predict the noise residual for prompt_a --> store the attentions in attention_store
                noise_pred_sub1 = self.retrieval_unet_sub1(
                    latent_model_input_sub1,
                    t,
                    encoder_hidden_states=prompt_embeds_a,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # predict the noise residual for prompt_b --> store the attentions in attention_store
                noise_pred_sub2 = self.retrieval_unet_sub2(
                    latent_model_input_sub2,
                    t,
                    encoder_hidden_states=prompt_embeds_b,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # predict the noise residual for the main prompt --> retrieve the attentions in retrieval_attention_store
                noise_pred_retrieved = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_main,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                cattn_img = show_cross_attention(
                    prompt=prompt,
                    attention_maps=retrieval_attention_store.global_store["down_cross"][
                        0
                    ],
                    tokenizer=self.tokenizer,
                    indices_to_alter=[
                        cfg.token_indices[0][0][0][0],
                        cfg.token_indices[1][0][0][0],
                    ],
                )
                os.makedirs(
                    cfg.output_path / cfg.prompt / "logs" / "prompt", exist_ok=True
                )
                cattn_img.save(
                    cfg.output_path / cfg.prompt / "logs" / "prompt" / f"attn_{i}.png"
                )

                cattn_img = show_cross_attention(
                    prompt=cfg.prompt_a,
                    attention_maps=attention_store_sub1.global_store["down_cross"][0],
                    tokenizer=self.tokenizer,
                    indices_to_alter=[
                        cfg.token_indices[0][1][0][0],
                    ],
                )
                os.makedirs(
                    cfg.output_path / cfg.prompt / "logs" / "prompt_a",
                    exist_ok=True,
                )
                cattn_img.save(
                    cfg.output_path / cfg.prompt / "logs" / "prompt_a" / f"attn_{i}.png"
                )

                cattn_img = show_cross_attention(
                    prompt=cfg.prompt_b,
                    attention_maps=attention_store_sub2.global_store["down_cross"][0],
                    tokenizer=self.tokenizer,
                    indices_to_alter=[
                        cfg.token_indices[1][1][0][0],
                    ],
                )
                os.makedirs(
                    cfg.output_path / cfg.prompt / "logs" / "prompt_b",
                    exist_ok=True,
                )
                cattn_img.save(
                    cfg.output_path / cfg.prompt / "logs" / "prompt_b" / f"attn_{i}.png"
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_retrieved.chunk(2)
                    noise_pred_main = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(
                #         noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                #     )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred_main,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_sub1.chunk(2)
                    noise_pred_sub1 = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents_sub1 = self.retrieval_scheduler_sub1.step(
                    noise_pred_sub1,
                    t,
                    latents_sub1,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_sub2.chunk(2)
                    noise_pred_sub2 = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents_sub2 = self.retrieval_scheduler_sub2.step(
                    noise_pred_sub2,
                    t,
                    latents_sub2,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # latents_sub1 = copy.deepcopy(latents)
                # latents_sub2 = copy.deepcopy(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            has_nsfw_concept = None
            # image, has_nsfw_concept = self.run_safety_checker(
            #     image, device, prompt_embeds_main.dtype
            # )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
