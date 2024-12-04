import os
from PIL import Image
from google.colab import userdata, drive, files
from datasets import load_dataset
import datasets
import torch
import shutil
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import OrderedDict
import peft
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL, AutoPipelineForText2Image, FluxPipeline
from diffusers.utils import make_image_grid
from string import whitespace
from safetensors.torch import load_file
import tensorflow as tf
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import requests
import sys
import prep_utils

prompts = [
    "Business headshot of ohwx with a neutral facial expression, wearing formal professional attire, against a light gray background with soft, natural lighting",
    "Professional portrait of ohwx with a warm smile, dressed in formal business wear, with a subtle gradient background and gentle, indirect lighting",
    "Formal headshot of ohwx with a slight smile, positioned according to the rule of thirds, against a crisp white background with high-contrast lighting",
    "Business photo of ohwx with a neutral expression, wearing a professional outfit with a shallow depth of field and a subtle bokeh effect in the background",
    "High-resolution portrait of ohwx with a friendly smile, dressed in formal business attire, with a subtle texture to the background and soft, diffused lighting",
    "Corporate headshot of ohwx with a professional expression, positioned in the center of the frame, against a plain dark gray background with dramatic, high-contrast lighting",
    "Modern business portrait of ohwx with a relaxed smile, wearing formal professional clothing, with a blurred background and a shallow depth of field",
    "Formal business photo of ohwx with a neutral expression, dressed in professional business wear, with a plain white background and soft, indirect lighting",
    "Professional headshot of ohwx with a warm smile, positioned off-center according to the rule of thirds, against a light blue background with a subtle gradient effect",
    "High-quality business portrait of ohwx with a confident expression, wearing formal professional attire, with a dark gray background and dramatic, high-contrast lighting"
    ]

textinv_prompts = [
    "Business headshot of <sks> with a neutral facial expression, wearing formal professional attire, against a light gray background with soft, natural lighting",
    "Professional portrait of <sks> with a warm smile, dressed in formal business wear, with a subtle gradient background and gentle, indirect lighting",
    "Formal headshot of <sks> with a slight smile, positioned according to the rule of thirds, against a crisp white background with high-contrast lighting",
    "Business photo of <sks> with a neutral expression, wearing a professional outfit with a shallow depth of field and a subtle bokeh effect in the background",
    "High-resolution portrait of <sks> with a friendly smile, dressed in formal business attire, with a subtle texture to the background and soft, diffused lighting",
    "Corporate headshot of <sks> with a professional expression, positioned in the center of the frame, against a plain dark gray background with dramatic, high-contrast lighting",
    "Modern business portrait of <sks> with a relaxed smile, wearing formal professional clothing, with a blurred background and a shallow depth of field",
    "Formal business photo of <sks> with a neutral expression, dressed in professional business wear, with a plain white background and soft, indirect lighting",
    "Professional headshot of <sks> with a warm smile, positioned off-center according to the rule of thirds, against a light blue background with a subtle gradient effect",
    "High-quality business portrait of <sks> with a confident expression, wearing formal professional attire, with a dark gray background and dramatic, high-contrast lighting"
    ]

neg_grid_prompt= "pixelated, oversatured, red faced, cross eyed, looking away, unnatural eyes, bad teeth , deformed, disfigured, distorted, blurry, bad anatomy, mutated, fused fingers, black and white, text, long neck, low resolution, bad proportions, unnatural body, old photo, vintage, poor facial details,"

def insert_string(prompts, subject, method):
  if method == 'textinv':
    return [prompt.replace("<sks>", f"<sks> {prep_utils.get_gender(subject)}") for prompt in prompts]
  else:
    return [prompt.replace("ohwx", f"ohwx {prep_utils.get_gender(subject)}") for prompt in prompts]

def get_inputs(prompts, neg_prompt, num_inference_steps, guidance_scale):
  if neg_prompt:
    return {
      "prompt": prompts,
      "neg_prompt":neg_prompt,
      "num_inference_steps": num_inference_steps,
      "guidance_scale": guidance_scale
    }
  else:
    return {
        "prompt": prompts,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }

def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    # Strip newline characters
    prompts = [prompt.strip() for prompt in prompts]
    return prompts

def save_image_set(image_set, save_dir, set_name):
  """Saves an image set to a specified directory.

  Args:
    image_set: A list of images to save.
    output_dir: The root directory where the set will be saved.
    set_name: The name of the subdirectory for the set.
  """

  # Create the subdirectory for the set
  set_dir = os.path.join(save_dir, f"infgrid_{set_name}_singles")
  os.makedirs(set_dir, exist_ok=True)

  # Save each image in the set
  for i, image in enumerate(image_set):
    image_path = os.path.join(set_dir, f'image_{i}.jpg')  # Adjust file format as needed
    image.save(image_path)

def save_inference_grid(save_dir, plots, subject):
   for i, plot in enumerate(plots):
    # counter= counter+1
    filepath = os.path.join(save_dir, f"Grid_{subject}_{i}")
    plot.savefig(filepath)

def create_image_grid(images, columns, rows, figsize=(10,10),labels_x=None, labels_y=None):
    """
    Creates a grid of images using matplotlib and numpy.

    Args:
        images: A list of image data.
        columns: The number of columns in the grid.
        rows: The number of rows in the grid.
        labels_x: A list of labels for the x-axis (columns).
        labels_y: A list of labels for the y-axis (rows).

    Returns:
        The figure and axes objects.
    """

    fig, axs = plt.subplots(rows, columns, figsize=(10, 10))  # Adjust figure size as needed

    for i in range(rows):
        for j in range(columns):
            index = i * columns + j
            if index < len(images):
                axs[i, j].imshow(images[index])
                axs[i, j].tick_params(axis='both', length=0, width=0)
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)

    # Add column labels
    if labels_x:
        for i in range(columns):
            axs[0, i].set_title(labels_x[i])

    # Add row labels
    if labels_y:
        for i in range(rows):
            axs[i, 0].set_ylabel(labels_y[i])

    return fig, axs 

def create_parameter_list(prompt, neg_prompt, grid_cols, grid_rows):
    """Creates a list of dictionaries containing the parameters.

    Args:
        prompt: The prompt for the model.
        neg_prompt: The negative prompt for the model.

    Returns:
        A list of dictionaries containing the parameters.
    """

    parameters = []
    if neg_prompt:
        for num_inference_steps in grid_cols:
            for guidance_scale in grid_rows:
                parameters.append({
                    "prompt": prompt,
                    "neg_prompt": neg_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                })
        return parameters
    else:
        for num_inference_steps in grid_cols:
            for guidance_scale in grid_rows:
                parameters.append({
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                })
        return parameters
#
# Inference Methods Below
#
def run_lora_sd15_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale, scheduler):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='lora')
  if checkpoint:
    checkpoint_dir = f"checkpoint-{checkpoint}"
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDv1.5'), subject), checkpoint_dir)
  else:
    model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'lora/SDv1.5'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionPipeline.from_pretrained(
    "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
  ).to("cuda")
  pipeline.unet.load_attn_procs(model_path)
  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  # prompts = read_prompts_from_file(prompt_path)
  inputs = get_inputs(gendered_prompts, neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images

def run_dlora_sd15_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale, scheduler):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='dlora')
  if checkpoint:
    checkpoint_dir = f"checkpoint-{checkpoint}"
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDv1.5'), subject), checkpoint_dir)
  else:
    model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dlora/SDv1.5'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionPipeline.from_pretrained(
    "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
  ).to("cuda")

  pipeline.unet.load_attn_procs(model_path)

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  inputs = get_inputs(gendered_prompts, neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images

def run_dbooth_sd15_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale, scheduler):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='dbooth')

  if checkpoint:
    checkpoint_dir = f"checkpoint-{checkpoint}"
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dbooth/SDv1.5'), subject), checkpoint_dir)
  else:
    model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dbooth/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dbooth/SDv1.5'), subject)
  os.makedirs(save_dir, exist_ok=True)

  if checkpoint:
    unet_path= os.path.join(model_path, "unet")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
      ).to("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
  else:
    pipeline = StableDiffusionPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  inputs = get_inputs(gendered_prompts,neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images

# only 500 step checkpoints possible (500, 1000, 1500 etc.)
def run_textinv_sd15_inference(subject, save_dir , checkpoint, num_inference_steps, guidance_scale, scheduler):
  textinv_gendered_prompts = insert_string(prompts=textinv_prompts, subject=subject, method='textinv')
  save_dir = os.path.join(os.path.join(save_dir, 'textinv/SDv1.5'), subject)
  os.makedirs(save_dir, exist_ok=True)

  if checkpoint:
    # checkpoint_dir = f"checkpoint-{checkpoint}"
    embed = f"learned_embeds-steps-{checkpoint}.safetensors"
    embed_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDv1.5'), subject), embed)
    pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
    pipeline.load_textual_inversion(embed_path, safety_checker=None, token="<sks>")

  else:
    raise ValueError("Checkpoint is required for this method.")
    # embed_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDv1.5'), subject) # picks latest embed(?)
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #   "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    #   ).to("cuda")
    # pipeline.load_textual_inversion(embed_path, safety_checker=None, token="<sks>")

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  inputs = get_inputs(textinv_gendered_prompts,neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images

def run_lora_sdxl_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale, scheduler):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='lora')
  if checkpoint:
    checkpoint_dir = f"checkpoint-{checkpoint}"
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDXL'), subject), checkpoint_dir)
  else:
    model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDXL'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'lora/SDXL'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, safety_checker=None,
  ).to("cuda")
  pipeline.load_lora_weights(model_path)
  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  # prompts = read_prompts_from_file(prompt_path)
  inputs = get_inputs(gendered_prompts,neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images


def run_dlora_sdxl_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale, scheduler):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='dlora')
  if checkpoint:
    checkpoint_dir = f"checkpoint-{checkpoint}"
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'SDXL'), subject), checkpoint_dir)
  else:
    model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDXL'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dlora/SDXL'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
  ).to("cuda")

  pipeline.load_lora_weights(model_path)

  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  inputs = get_inputs(gendered_prompts,neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images

# only 500 step checkpoints possible (500, 1000, 1500 etc.)
def run_textinv_sdxl_inference(subject, save_dir , checkpoint, num_inference_steps, guidance_scale, scheduler):
  textinv_gendered_prompts = insert_string(prompts=textinv_prompts, subject=subject, method='textinv')

  save_dir = os.path.join(os.path.join(save_dir, 'textinv/SDXL'), subject)
  os.makedirs(save_dir, exist_ok=True)

  if checkpoint:
    # checkpoint_dir = f"checkpoint-{checkpoint}"
    embed_1 = f"learned_embeds-steps-{checkpoint}.safetensors"
    embed_1_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_1)

    embed_2 = f"learned_embeds_2-steps-{checkpoint}.safetensors"
    embed_2_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_2)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")

    pipeline.load_textual_inversion(embed_1_path, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer, token="<sks>")
    pipeline.load_textual_inversion(embed_2_path, text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2, token="<sks>")

  else:
    embed_1 = f"learned_embeds.safetensors"
    embed_1_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_1) # picks latest embed(?)
    
    embed_2 = f"learned_embeds_2.safetensors"
    embed_2_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_2)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
    pipeline.load_textual_inversion(embed_1_path, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer, token="<sks>")
    pipeline.load_textual_inversion(embed_2_path, text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2, token="<sks>")

  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config) # fix
  pipeline.enable_attention_slicing()
  pipeline.enable_vae_slicing()
  pipeline.enable_model_cpu_offload()

  inputs = get_inputs(textinv_gendered_prompts,neg_grid_prompt, num_inference_steps, guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}_{scheduler}.png"))

  return images
#
#
# GRID METHODS BELOW
#
def generate_inference_grid_textinv_sd15(subject, save_dir, checkpoint, prompt, scheduler):
  save_dir = os.path.join(os.path.join(save_dir, 'textinv/SDv1.5/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  if checkpoint:
    # checkpoint_dir = f"checkpoint-{checkpoint}"
    embed = f"learned_embeds-steps-{checkpoint}.safetensors"
    embed_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDv1.5'), subject), embed)
    pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
    pipeline.load_textual_inversion(embed_path, safety_checker=None, token="<sks>")

  else:
    raise ValueError("Checkpoint is required for this method.")
    # embed_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDv1.5'), subject) # picks latest embed(?)
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #   embed_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    #   ).to("cuda")
    # pipeline.load_textual_inversion(embed_path, safety_checker=None, token="<sks>")

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_dbooth_sd15(subject, save_dir, checkpoint, prompt, scheduler):
  if checkpoint:
      checkpoint_dir = f"checkpoint-{checkpoint}"
      model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dbooth/SDv1.5'), subject), checkpoint_dir)
  else:
      model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dbooth/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dbooth/SDv1.5/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  # pipeline = StableDiffusionPipeline.from_pretrained(
  #     "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
  #   ).to("cuda")

  if checkpoint:
    unet_path= os.path.join(model_path, "unet")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
      ).to("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
  else:
    pipeline = StableDiffusionPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_lora_sd15(subject, save_dir, checkpoint, prompt, scheduler):
  if checkpoint:
      checkpoint_dir = f"checkpoint-{checkpoint}"
      model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDv1.5'), subject), checkpoint_dir)
  else:
      model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'lora/SDv1.5/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")
  pipeline.unet.load_attn_procs(model_path)
  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_dlora_sd15(subject, save_dir, checkpoint, prompt, scheduler):
  if checkpoint:
      checkpoint_dir = f"checkpoint-{checkpoint}"
      model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDv1.5'), subject), checkpoint_dir)
  else:
      model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDv1.5'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dlora/SDv1.5/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionPipeline.from_pretrained(
      "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")
  pipeline.unet.load_attn_procs(model_path)
  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_lora_sdxl(subject, save_dir, checkpoint, prompt, scheduler):
  if checkpoint:
      checkpoint_dir = f"checkpoint-{checkpoint}"
      model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDXL'), subject), checkpoint_dir)
  else:
      model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/SDXL'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'lora/SDXL/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")
  pipeline.load_lora_weights(model_path)

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)
  
  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_dlora_sdxl(subject, save_dir, checkpoint, prompt, scheduler):
  if checkpoint:
      checkpoint_dir = f"checkpoint-{checkpoint}"
      model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDXL'), subject), checkpoint_dir)
  else:
      model_path = os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'dlora/SDXL'), subject)

  save_dir = os.path.join(os.path.join(save_dir, 'dlora/SDXL/infgrid'), subject)
  os.makedirs(save_dir, exist_ok=True)

  pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
    ).to("cuda")

  pipeline.load_lora_weights(model_path)
  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         neg_prompt=neg_grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots

def generate_inference_grid_textinv_sdxl(subject, save_dir, checkpoint, prompt, scheduler):
  save_dir = os.path.join(os.path.join(save_dir, 'textinv/SDXL'), subject)
  os.makedirs(save_dir, exist_ok=True)

  if checkpoint:
      # checkpoint_dir = f"checkpoint-{checkpoint}"
      embed_1 = f"learned_embeds-steps-{checkpoint}.safetensors"
      embed_1_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_1)

      embed_2 = f"learned_embeds_2-steps-{checkpoint}.safetensors"
      embed_2_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_2)

      pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")

      pipeline.load_textual_inversion(embed_1_path, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer, token="<sks>")
      pipeline.load_textual_inversion(embed_2_path, text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2, token="<sks>")

  else:
    embed_1 = f"learned_embeds.safetensors"
    embed_1_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_1) # picks latest embed(?)
    
    embed_2 = f"learned_embeds_2.safetensors"
    embed_2_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'textinv/SDXL'), subject), embed_2)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
      ).to("cuda")
    pipeline.load_textual_inversion(embed_1_path, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer, token="<sks>")
    pipeline.load_textual_inversion(embed_2_path, text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2, token="<sks>")

  pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_vae_slicing()
  # pipeline.enable_model_cpu_offload()
  pipeline.enable_attention_slicing()

  vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
  pipeline.vae = vae

  grid_prompt = prompt
  

  grid_cols = np.linspace(20, 50, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(6, 9, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(grid_prompt, neg_grid_prompt, grid_cols, grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                          neg_prompt=neg_grid_prompt,
                          num_inference_steps=line["num_inference_steps"],
                          guidance_scale=line["guidance_scale"]).images
                      )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)
  
  return images, plots

#
# Flux Inference
#
# can only take checkpoints from 2200, 2400, 2600, 2800, as that is what i saved
def run_lora_flux_inference(subject, save_dir, checkpoint, num_inference_steps, guidance_scale):
  gendered_prompts = insert_string(prompts=prompts, subject=subject, method='lora')

  save_dir = os.path.join(os.path.join(save_dir, 'lora/flux'), subject)
  os.makedirs(save_dir, exist_ok=True)
  
  pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

  if checkpoint:
    lora_checkpoint_weight = f"{subject}_00000{str(checkpoint).zfill(4)}" #TODO: not sure if this works correctly at all times 
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/flux'), subject), subject)
    pipeline.load_lora_weights(model_path, weight_name=lora_checkpoint_weight)
  else:
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/flux'), subject), subject)
    lora_weight = f"{subject}.safetensors"
    pipeline.load_lora_weights(model_path, weight_name=lora_weight)

  # vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev")
  # pipeline.vae = vae

  # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
  pipeline.enable_sequential_cpu_offload()
  pipeline.vae.enable_slicing()
  pipeline.vae.enable_tiling()
  pipeline.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

  inputs = get_inputs(prompts=gendered_prompts,neg_prompt=None, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
  images = pipeline(**inputs).images # amount of images generated is dependent on number of prompts

  # Save each image to the specified directory
  for i, image in enumerate(images):
      image.save(os.path.join(save_dir, f"{subject}_prompt{i}_{num_inference_steps}_{guidance_scale}.png"))

  return images
#
# Flux Inference Grid
#
def generate_inference_grid_lora_flux(subject, save_dir, checkpoint, prompt):
  save_dir = os.path.join(os.path.join(save_dir, 'lora/flux'), subject)
  os.makedirs(save_dir, exist_ok=True)
  
  pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, use_safetensors=True)

  if checkpoint:
    lora_checkpoint_weight = f"{subject}_00000{str(checkpoint).zfill(4)}" #TODO: not sure if this works correctly at all times 
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/flux'), subject), subject)
    pipeline.load_lora_weights(model_path, weight_name=lora_checkpoint_weight)
  else:
    model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/flux'), subject), subject)
    lora_weight = f"{subject}.safetensors"
    pipeline.load_lora_weights(model_path, weight_name=lora_weight)
    #####
    # lora_weight = f"{subject}.safetensors"
    # model_path = os.path.join(os.path.join(os.path.join(os.environ.get('MODELS_PARENT_DIR'), 'lora/FLUX'), subject), subject)
    # pipeline.load_lora_weights(model_path)

  # vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev")
  # pipeline.vae = vae

  # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
  pipeline.enable_sequential_cpu_offload()
  pipeline.vae.enable_slicing()
  pipeline.vae.enable_tiling()
  pipeline.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

  grid_prompt = prompt

  grid_cols = np.linspace(30, 60, 4, endpoint=True, dtype=int)
  grid_rows = np.linspace(3, 6, 4, endpoint=True, dtype=int)

  parameter_list = create_parameter_list(prompt=grid_prompt,neg_prompt=None, grid_cols=grid_cols, grid_rows=grid_rows)

  inference_steps_list = my_list = list(OrderedDict.fromkeys([param["num_inference_steps"] for param in parameter_list]).keys())
  guidance_scale_list = my_list = list(OrderedDict.fromkeys([param["guidance_scale"] for param in parameter_list]).keys())
  labels_x = [f"Guidance Scale: {step}" for step in guidance_scale_list]
  labels_y = [f"Inference Steps: {step}" for step in inference_steps_list]

  images = []
  for line in parameter_list:
      images.extend(pipeline(prompt=grid_prompt,
                         num_inference_steps=line["num_inference_steps"],
                         guidance_scale=line["guidance_scale"]).images
                    )

  plots = []
  fig, axs = create_image_grid(images, columns=len(grid_cols),rows=len(grid_rows), labels_x=labels_x, labels_y=labels_y)
  plt.subplots_adjust(hspace=0, wspace=0)
  plots.append(fig)

  save_inference_grid(save_dir= save_dir, plots=plots, subject=subject)
  save_image_set(images, save_dir= save_dir, set_name=subject)

  return images, plots