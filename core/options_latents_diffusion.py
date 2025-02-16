import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    gradient_checkpointing: bool = False
    enable_xformers_memory_efficient_attention: bool = False
    pretrained_model_name_or_path: str = "/remote-home1/yeyang/aigc/model/stable-diffusion-v1-5"
    input_size: int = 256
    input_ray_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    data_path: str = '/remote-home1/yeyang/aigc/dataset2'
    json_path:str = '/remote-home1/yeyang/aigc/dataset1'
    # fovy of the dataset
    fovy: float = 39.6
    # camera near plane
    znear: float = 0.01
    # camera far plane
    zfar: float = 1000
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 16
    snr_gamma: int = 5
    ### training
    # workspace
    workspace: str = './workspace'
    workspace1: Optional[str] = None
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0 ##TZY
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 5e-5
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    checkpoints_total_limit: int = 3

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

# config_doc['lrm'] = 'the default settings for LGM'
# config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=64,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=32,
    output_size=64, # render & supervise Gaussians at a higher resolution.
    batch_size=96,
    num_views=10,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big_latent'] = 'big model with higher resolution Gaussians'
config_defaults['big_latent'] = Options(
    input_size=64,
    down_channels=(256, 512, 1024, 1024),
    down_attention=(True, True, True, False),
    up_channels=(1024, 1024, 512, 256),
    up_attention=(False, True, True, True),
    splat_size=64,
    output_size=64, # render & supervise Gaussians at a higher resolution.
    batch_size = 2, # 2
    num_views= 8,
    gradient_accumulation_steps= 6, # 16
    mixed_precision='bf16',
)

config_doc['big_latent_sd'] = 'big model with higher resolution Gaussians'
config_defaults['big_latent_sd'] = Options(
    gradient_checkpointing = True,
    enable_xformers_memory_efficient_attention = True,
    lr = 1e-4, 
    #lambda_lpips = 0.5,
    lambda_lpips = 2,
    input_size=64,
    down_channels=(256, 512, 1024, 1024),
    down_attention=(True, True, True, False),
    up_channels=(1024, 1024, 512, 256),
    up_attention=(False, True, True, True),
    splat_size=64,
    output_size=64, # render & supervise Gaussians at a higher resolution.
    batch_size = 2, # 2
    num_views= 8,
    gradient_accumulation_steps= 6, # 16
    mixed_precision='bf16',
)
config_defaults['big_latent_sd_diffusion'] = Options(
    gradient_checkpointing = True,
    enable_xformers_memory_efficient_attention = True,
    lr = 1e-4, 
    lambda_lpips = 0.5,
    #lambda_lpips = 2,
    input_size=64,
    down_channels=(256, 512, 1024, 1024),
    down_attention=(True, True, True, False),
    up_channels=(1024, 1024, 512, 256),
    up_attention=(False, True, True, True),
    splat_size=64,
    output_size=64, # render & supervise Gaussians at a higher resolution.
    batch_size = 2, # 2
    num_views= 8,
    gradient_accumulation_steps= 2, # 16
    mixed_precision='bf16',
    num_epochs = 50,
)

config_defaults['big_latent_sd_diffusion_insert'] = Options(
    gradient_checkpointing = True,
    enable_xformers_memory_efficient_attention = True,
    lr = 1e-4, 
    lambda_lpips = 0.5,
    #lambda_lpips = 2,
    input_size=64,
    #resume= "/remote-home1/yeyang/aigc/models/models--ashawkey--LGM/snapshots/1c28a2fd3bb1982414f722503ae862bdbb82636c/model_fp16_fixrot.safetensors",
    resume= 'workspace_1e-4_latent_diffusion_unet_LGM_insert3/model.safetensors',
    up_channels=(1024, 1024, 512, 256, 128),
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size= 512, # render & supervise Gaussians at a higher resolution.
    batch_size = 8, # 2
    num_views= 8,
    gradient_accumulation_steps= 1, # 16
    mixed_precision='bf16',
)

config_defaults['big_latent_sd_diffusion_compose'] = Options(
    gradient_checkpointing = True,
    enable_xformers_memory_efficient_attention = True,
    lr = 1e-4, 
    lambda_lpips = 0.5,
    #lambda_lpips = 2,
    input_size=64,
    resume= "/remote-home1/yeyang/aigc/models/models--ashawkey--LGM/snapshots/1c28a2fd3bb1982414f722503ae862bdbb82636c/model_fp16_fixrot.safetensors",
    #resume= 'workspace_1e-4_latent_diffusion_unet_LGM_compose_text/model.safetensors',
    up_channels=(1024, 1024, 512, 256, 128),
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size= 512, # render & supervise Gaussians at a higher resolution.
    batch_size = 8, # 2
    num_views= 8,
    gradient_accumulation_steps= 1, # 16
    mixed_precision='bf16',
)

config_doc['big_latent_lpips'] = 'big model with higher resolution Gaussians'
config_defaults['big_latent_lpips'] = Options(
    input_size=64,
    down_channels=(256, 512, 1024, 1024),
    down_attention=(True, True, True, False),
    up_channels=(1024, 1024, 512, 256),
    up_attention=(False, True, True, True),
    splat_size=64,
    output_size=64, # render & supervise Gaussians at a higher resolution.
    batch_size=6, # 2
    num_views=10,
    gradient_accumulation_steps=16, # 16
    mixed_precision='bf16',
)

# config_doc['big_latent'] = 'big model with higher resolution Gaussians'
# config_defaults['big_latent'] = Options(
#     input_size=64,
#     down_channels=(256, 512, 1024, 1024),
#     down_attention=(True, True, True, False),
#     up_channels=(1024, 1024, 512, 256),
#     up_attention=(False, True, True, True),
#     splat_size=64,
#     output_size=64, # render & supervise Gaussians at a higher resolution.
#     batch_size=15, # 2
#     num_views=10,
#     gradient_accumulation_steps=4, # 16
#     mixed_precision='bf16',
# )

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
