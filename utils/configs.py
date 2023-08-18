from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from utils.ptp_utils import Pharse2idx
import ast


@dataclass
class ComposableDiffusionConfig:
    # Guiding text prompt
    prompt: str = None
    # Sub-prompt #1
    prompt_a: str = None
    # Sub-prompt #2
    prompt_b: str = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path("./outputs/composable_diffusion")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class LayoutGuidanceConfig:
    # Guiding text prompt
    prompt: str = "A hello kitty toy is playing with a purple ball."
    # Provide the bounding box
    bounding_box: str = "[[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]"
    # Provide the phrases
    phrases: str = "hello kitty;ball"
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path("./outputs/layout_guidance")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # attention_aggregation_method, avaliable methods are aggregate_attention, all_attention, aggregate_layer_attention
    attention_aggregation_method: str = "all_attention"
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_backward: int = 10
    # Loss threshold
    loss_threshold: float = 0.2
    # Loss scale
    loss_scale: float = 30
    # Max iterations per step
    max_iter_per_step: int = 5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or Layout-guidacne
    run_standard_sd: bool = False

    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 30
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))

    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = False
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3

    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.bounding_box = ast.literal_eval(self.bounding_box)

        assert self.attention_aggregation_method in [
            "all_attention",
            "aggregate_attention",
            "aggregate_layer_attention",
        ], "Invalid attention aggregation method supported types are: `['all_attention','aggregate_attention','aggregate_layer_attention']`"

    @property
    def bbox(self):
        tmp_ = {}
        for k, v in zip(self.bbox_index, self.bounding_box):
            tmp_[k] = v
        return tmp_

    @property
    def object_positions(self):
        return Pharse2idx(self.prompt, self.bbox_phrases)

    @property
    def bbox_phrases(self):
        return self.phrases.split(";")


@dataclass
class AttentionRefocusConfig:
    # Guiding text prompt
    prompt: str = "A hello kitty toy is playing with a purple ball."
    # Provide the bounding box
    bounding_box: str = "[[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]"
    # Provide the phrases
    phrases: str = "hello kitty;ball"
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path("./outputs/attention_refocus")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # attention_aggregation_method, avaliable methods are aggregate_attention, all_attention, aggregate_layer_attention
    attention_aggregation_method: str = "aggregate_attention"
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_backward: int = 10
    # Loss threshold
    loss_threshold: float = 0.1
    # Loss scale
    loss_scale: float = 10
    # Max iterations per step
    max_iter_per_step: int = 5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or Layout-guidacne
    run_standard_sd: bool = False

    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 10
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))

    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = False
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3

    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.bounding_box = ast.literal_eval(self.bounding_box)

    @property
    def bbox(self):
        tmp_ = {}
        for k, v in zip(self.bbox_index, self.bounding_box):
            tmp_[k] = v
        return tmp_

    @property
    def object_positions(self):
        return Pharse2idx(self.prompt, self.bbox_phrases)

    @property
    def bbox_phrases(self):
        return self.phrases.split(";")


@dataclass
class AttendExciteConfig:
    # Guiding text prompt
    prompt: str = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path("./outputs/attend_excite")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8}
    )
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str = None
    inference_outdir: Path = Path("outputs/test_images")
    prompt: str = None

    def __post_init__(self):
        self.inference_outdir.mkdir(exist_ok=True, parents=True)


@dataclass
class TrainerConfig:
    pretrained_model_name_or_path: str = None
    revision: str = None
    tokenizer_name: str = None
    max_train_steps: int = None
    regularizer: str = "lg"

    seed: int = None
    output_dir: Path = Path("./outputs/text-inversion-model")
    resolution: int = 512
    center_crop: bool = False
    train_text_encoder: bool = True
    train_batch_size: int = 4  # per device
    sample_batch_size: int = 4  # per device
    num_train_epochs: int = 1
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = None
    resume_from_checkpoint: str = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_8bit_adam: bool = False
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    push_to_hub: bool = False
    hub_token: str = None
    hub_model_id: str = None
    logging_dir: Path = Path("./logs")
    allow_tf32: bool = False
    report_to: str = "wandb"
    validation_prompt: str = None
    num_validation_images: int = 4
    validation_steps: int = 100
    mixed_precision: str = "no"
    prior_generation_precision: str = "none"
    local_rank: int = -1
    enable_xformers_memory_efficient_attention: bool = False
    set_grads_to_none: bool = False
    offset_noise: bool = False
    pre_compute_text_embeddings: bool = False
    tokenizer_max_length: int = None
    text_encoder_use_attention_mask: bool = False
    skip_save_text_encoder: bool = False
    validation_images = None
    class_labels_conditioning = None

    ## dreambooth specific parameters
    instance_data_dir: str = None
    class_data_dir: str = None
    instance_prompt: str = None
    class_prompt: str = None
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logging_dir.mkdir(exist_ok=True, parents=True)

        # self.output_dir = str(self.output_dir)
        # self.logging_dir = str(self.logging_dir)
