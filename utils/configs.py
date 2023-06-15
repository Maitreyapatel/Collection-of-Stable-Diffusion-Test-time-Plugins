from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


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
    output_path: Path = Path('./outputs')
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

@dataclass
class InferenceConfig:
    loss_scale: int = 30
    batch_size: int  = 1
    loss_threshold: float = 0.2
    max_iter: int = 5
    max_index_step: int = 10
    timesteps: int = 51
    classifier_free_guidance: float = 7.5
    rand_seed: int = 400

@dataclass
class NoiseScheduleConfig:
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    num_train_timesteps: int = 1000

@dataclass
class paths:
    save_path: str = '/home/ovengurl/LSDGen/outputs'
    model_path: str = 'runwayml/stable-diffusion-v1-5'
    unet_config: str = '/home/ovengurl/LSDGen/utils/lg_unet.json'
    device_ids: List[int] = field(default_factory=lambda: [0,1,2,3])

@dataclass
class LayoutGuidanceConfig:
    prompt: str = None
    phrases: str = None
    # get bounding boxes from the user for example [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]
    bboxes: List[List[List[float]]] = None
    general: paths = paths
    inference: InferenceConfig = InferenceConfig
    noise_schedule: NoiseScheduleConfig = NoiseScheduleConfig
