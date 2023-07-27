import os
import json

import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel

def run_inference(args):
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=torch.float32, safety_checker=None
        ).to("cuda")
    except:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, safety_checker=None
        ).to("cuda")

        text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
        )
        text_encoder.load_state_dict(torch.load(f"/data_5/data/matt/LSDGen/{args.pretrained_model_name_or_path}/checkpoint-1000/text_encoder/pytorch_model.bin"))
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        )
        unet.load_state_dict(torch.load(f"/data_5/data/matt/LSDGen/{args.pretrained_model_name_or_path}/checkpoint-1000/unet/diffusion_pytorch_model.bin"))
    
        pipe.unet = unet.cuda()
        pipe.text_encoder = text_encoder.cuda()

    
    prompts = [args.prompt]*4
    images = pipe(
        prompts,
        num_inference_steps=100,
        guidance_scale=10.0,
    ).images

    for en, (img, prompt) in enumerate(zip(images, prompts)):
        img.save(os.path.join(args.inference_outdir, f"{prompt.replace(' ', '_').replace('.','')}_{en}.jpeg"))

    return