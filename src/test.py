import torch
from diffusers import StableDiffusionPipeline
import json
import os


def run_inference(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.float32, safety_checker=None
    ).to("cuda")

    prompts = [args.prompt]*4
    images = pipe(
        prompts,
        num_inference_steps=100,
        guidance_scale=10.0,
    ).images

    for en, (img, prompt) in enumerate(zip(images, prompts)):
        img.save(os.path.join(args.inference_outdir, f"{prompt.replace(' ', '_').replace('.','')}_{en}.jpeg"))

    return