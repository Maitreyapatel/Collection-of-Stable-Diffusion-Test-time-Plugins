from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from utils.ptp_utils import Pharse2idx
import ast

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
    output_path: Path = Path('./outputs/layout_guidance')
    # Number of denoising steps
    n_inference_steps: int = 100
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

    @property
    def bbox(self):
        tmp_ = {}
        for k,v in zip(self.bbox_index, self.bounding_box):
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
    output_path: Path = Path('./outputs/layout_guidance')
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
        for k,v in zip(self.bbox_index, self.bounding_box):
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
    output_path: Path = Path('./outputs/layout_guidance')
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
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
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
