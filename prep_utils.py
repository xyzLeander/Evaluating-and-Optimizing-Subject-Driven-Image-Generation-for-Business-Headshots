
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


def get_output_dir(results_dir, training_method, training_model, training_subject):
  output_dir = os.path.join(results_dir, training_method, training_model, training_subject)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  os.environ['OUTPUT_DIR'] = output_dir # set env variable where to save model
  return output_dir

def get_token(training_method):
  if training_method == 'lora' or training_method == 'dlora' or training_method =='dbooth':
    token = 'ohwx'
    return token
  elif training_method =='textinv':
    token = '<sks>'
    return token
  else:
    raise ValueError("Token cannot be set automatically for this method. Make sure training method is spelled correctly.")

def get_gender(training_subject):
  male_list = ['timh', 'timk', 'leander', 'patrick', 'christoph', 'jannik', 'marco', 'nils']
  female_list = ['hannah', 'celine']

  if training_subject in male_list:
    gender = 'man'
    return gender
  elif training_subject in female_list:
    gender = 'woman'
    return gender
  else:
    raise ValueError("Gender cannot be set automatically for this subject. Make sure subject name is spelled correctly, or adjust get_gender method to include new name.")
## valid_prompt gets automatically set in setup methode already

# def get_validation_prompt(training_method ,training_subject):
#   male_list = ['timh', 'timk', 'leander', 'patrick', 'christoph', 'jannik', 'marco', 'nils']
#   female_list = ['hannah', 'celine']

#   if training_subject in male_list:
#     if training_method == 'lora' or training_method == 'dlora' or training_method == 'dbooth':
#       print("Training subject is set as male. Validation prompt is adjusted accordingly.")
#       prompt = "business headshot of a ohwx male"
#       return prompt
#     elif training_method == 'textinv':
#       print("Training subject is set as male. Validation prompt is adjusted accordingly.")
#       prompt = "business headshot of a <sks> male"
#       return prompt
#   elif training_subject in female_list:
#     if training_method == 'lora' or training_method == 'dlora' or training_method == 'dbooth':
#       print("Training subject is set as female. Validation prompt is adjusted accordingly.")
#       prompt = "business headshot of a ohwx female"
#       return prompt
#     elif training_method == 'textinv':
#       print("Training subject is set as female. Validation prompt is adjusted accordingly.")
#       prompt = "business headshot of a <sks> female"
#       return prompt
#   else:
#     raise ValueError("Validation prompt cannot be set automatically for this subject. Make sure subject name is spelled correctly, or adjust get_validation_prompt method to include new name.")

def load_images(base_dir, training_subject):
    images = []

    # Construct the target subfolder name
    target_subfolder = f"{training_subject}_10"

    # Traverse the base directory
    for root, dirs, files in os.walk(base_dir):
        if target_subfolder in dirs:
            # Construct the full path to the target subfolder
            target_subfolder_path = os.path.join(root, target_subfolder)
            for file in os.listdir(target_subfolder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(target_subfolder_path, file)
                    image = Image.open(image_path)
                    images.append(image)
                    # print(f"Loaded {image_path}")
            break  # Exit the loop once the target subfolder is found and images are loaded

    if not images:
        print(f"Subfolder '{target_subfolder}' not found in base directory '{base_dir}'")

    return images

# def move_images(base_name, images_dict, train_directory):
#   if base_name in images_dict:
#       images = images_dict[base_name]

#       # Create the new directory if it doesn't exist
#       if not os.path.exists(train_directory):
#           os.makedirs(train_directory)

#       for i, image in enumerate(images):
#           output_path = os.path.join(train_directory, f'{base_name}_{i+1}.jpg')
#           image.save(output_path)
#           print(f"Saved {output_path}")
#   else:
#       print(f"No images found for base name: {base_name}")

def move_images(base_name, images, train_directory):
    # Create the new directory if it doesn't exist
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)

    if images:
        for i, image in enumerate(images):
            output_path = os.path.join(train_directory, f'{base_name}_{i+1}.jpg')
            image = image.convert('RGB')
            image.save(output_path)
            print(f"Saved {output_path}")
    else:
        print(f"No images found for base name: {base_name}")

def empty_directory(train_directory):
    # Check if the directory exists
    if not os.path.exists(train_directory):
        print(f"Directory does not exist: {train_directory}")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(train_directory):
        file_path = os.path.join(train_directory, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
        else:
            print(f"Skipped non-file: {file_path}")

    print(f"Directory {train_directory} has been emptied.")
  
def create_folder_structure(root_dir, subfolders):
  # Mount Google Drive
  drive.mount('/content/drive')

  # List of sub-subfolders
  subsubfolders = ['SDXL', 'SDv1.5', 'flux']

  # Create the root directory if it doesn't exist
  if not os.path.exists(root_dir):
      os.makedirs(root_dir)
      print(f"Created root directory: {root_dir}")
  else:
      print(f"Root directory already exists: {root_dir}")

  # Create the folder structure
  for subfolder in subfolders:
      subfolder_path = os.path.join(root_dir, subfolder)
      if not os.path.exists(subfolder_path):
          os.makedirs(subfolder_path)
          print(f"Created subfolder: {subfolder_path}")
      else:
          print(f"Subfolder already exists: {subfolder_path}")

      for subsubfolder in subsubfolders:
          subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
          if not os.path.exists(subsubfolder_path):
              os.makedirs(subsubfolder_path)
              print(f"Created sub-subfolder: {subsubfolder_path}")
          else:
              print(f"Sub-subfolder already exists: {subsubfolder_path}")

  print("Folder structure creation completed!")

def alt_upload_images(train_directory):
  print('Uploading images to ' + train_directory)
  uploaded = files.upload()
  for filename in uploaded.keys():
      dst_path = os.path.join(train_directory, filename)
      shutil.move(filename, dst_path)

def caption_images(train_directory, text_conditioning):
  # Collect all file names and their paths
  file_names = []
  for root, dirs, files in os.walk(train_directory):
      for file in files:
          if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
              file_names.append(os.path.join(root, file))

  # Load the dataset
  dataset = load_dataset("imagefolder", data_dir=train_directory, split="train")

  # Set up the captioning processor and model
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

  # Define the JSONL file path
  jsonl_file_path = os.path.join(train_directory, "metadata.jsonl")

# Check if the file already exists and delete it
  if os.path.exists(jsonl_file_path):
      os.remove(jsonl_file_path)
      print(f"File {jsonl_file_path} already exists and has been deleted.")

  # Create and write to the JSONL file
  with open(jsonl_file_path, "w") as f:
      for i, file_name in enumerate(file_names):
          # Load the image from the file path
          image = Image.open(file_name).convert("RGB")

          # Generate caption
          inputs = processor(image, text_conditioning, return_tensors="pt").to("cuda")
          out = model.generate(**inputs)
          caption = processor.decode(out[0], skip_special_tokens=True)

          # Write to JSONL file
          file_data = {
              "file_name": os.path.basename(file_name),
              "text": caption
          }
          f.write(json.dumps(file_data) + "\n")
  print("Captioning completed at " + jsonl_file_path)
  # clear up memory
  print("Memory before deleting captioning processor & model")
  memory_stats()
  del processor
  del model
  torch.cuda.empty_cache()
  print("Memory after deleting captioning processor & model")
  memory_stats()

def memory_stats():
  print(torch.cuda.memory_allocated()/1024**2)
  print(torch.cuda.memory_reserved()/1024**2)

def delete_pipeline(pipe):
  # del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2, pipe.unet, pipe.vae
  print("Memory before deleting pipeline components")
  memory_stats()
  
  del pipe.unet, pipe.vae, pipe.scheduler, pipe.tokenizer, pipe.text_encoder
  del pipe
  torch.cuda.empty_cache()
  print("Memory after deleting pipeline components")
  memory_stats()

def setup_training_environment(train_directory, results_dir, training_model, training_method, training_subject):
  # Get token and gender based on training method and subject
  output_dir = get_output_dir(results_dir, training_method, training_model, training_subject)
  token = get_token(training_method)
  gender = get_gender(training_subject)

  # Set environment variables for training directory and output directory
  os.environ['TRAIN_DIR'] = train_directory
  os.environ['OUTPUT_DIR'] = output_dir

  # Set environment variables based on the training model
  if training_model == 'SDv1.5':
      os.environ['MODEL_NAME'] = 'benjamin-paine/stable-diffusion-v1-5'
      os.environ['RESOLUTION'] = '512'
  elif training_model == 'SDXL':
      os.environ['MODEL_NAME'] = 'stabilityai/stable-diffusion-xl-base-1.0'
      os.environ['VAE_NAME'] = 'madebyollin/sdxl-vae-fp16-fix'
      os.environ['RESOLUTION'] = '1024'
  elif training_model == 'flux':
      os.environ['MODEL_NAME'] = 'black-forest-labs/FLUX.1-dev'  # Placeholder for the actual model name
      os.environ['TRIGGER_WORD'] = token
      os.environ['GENDER'] = gender
  else:
      raise ValueError('training_model variable cannot be resolved')

  # Set environment variables based on the training method
  if training_method == 'dbooth' or training_method == 'dlora':
      os.environ['CLASS_PROMPT'] = f"photo of a {gender}"
      os.environ['INSTANCE_PROMPT'] = f"photo of a {token} {gender}"
      os.environ['VALID_PROMPT'] = f"business headshot of a {token} {gender}"


      class_dir = os.path.join(os.path.dirname(train_directory), gender)
      os.makedirs(class_dir, exist_ok=True)
      os.environ['CLASS_DIR'] = class_dir
  elif training_method == 'lora':
      os.environ['VALID_PROMPT'] = f"business headshot of a {token} {gender}"
      os.environ['INSTANCE_PROMPT'] = f"photo of a {token} {gender}"
  elif training_method == 'textinv':
      os.environ['PLACEHOLDER_TOKEN'] = token
      os.environ['INITIALIZER_TOKEN'] = gender
      os.environ['VALID_PROMPT'] = f"business headshot of a {token} {gender}"
  else:
      raise ValueError('training_method variable cannot be resolved')

#TODO: adjust paths according to subject etc.
def create_job_config(name, max_steps, save_steps, lr,train_directory, valid_steps):
  job_config = OrderedDict([
      ('job', 'extension'),
      ('config', OrderedDict([
          # this name will be the folder and filename name
          ('name', name),
          ('process', [
              OrderedDict([
                  ('type', 'sd_trainer'),
                  ('training_folder', os.environ.get("OUTPUT_DIR")),
                  ('device', 'cuda:0'),
                  ('trigger_word', os.environ.get("TRIGGER_WORD")),
                  ('network', OrderedDict([
                      ('type', 'lora'),
                      ('linear', 16),
                      ('linear_alpha', 16)
                  ])),
                  ('save', OrderedDict([
                      ('dtype', 'float16'),  # precision to save
                      ('save_every', save_steps),  # save every this many steps
                      ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
                  ])),
                  ('datasets', [
                      OrderedDict([
                          ('folder_path', train_directory),
                          ('caption_ext', 'txt'),
                          ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
                          ('shuffle_tokens', False),  # shuffle caption order, split by commas
                          ('cache_latents_to_disk', True),  # leave this true unless you know what you're doing
                          ('resolution', [512, 768, 1024])  # flux enjoys multiple resolutions
                      ])
                  ]),
                  ('train', OrderedDict([
                      ('batch_size', 1),
                      ('steps', max_steps),  # total number of steps to train 500 - 4000 is a good range
                      ('gradient_accumulation_steps', 1),
                      ('train_unet', True),
                      ('train_text_encoder', False),  # probably won't work with flux
                      ('content_or_style', 'balanced'),  # content, style, balanced
                      ('gradient_checkpointing', True),  # need the on unless you have a ton of vram
                      ('noise_scheduler', 'flowmatch'),  # for training only
                      ('optimizer', 'adamw8bit'),
                      ('lr', lr), #default= 1e-4 ?
                      ('ema_config', OrderedDict([
                          ('use_ema', True),
                          ('ema_decay', 0.99)
                      ])),
                      ('dtype', 'bf16')
                  ])),
                  ('model', OrderedDict([
                      # huggingface model name or path
                      ('name_or_path', "black-forest-labs/FLUX.1-dev"),
                      ('is_flux', True),
                      ('quantize', True),  # run 8bit mixed precision
                  ])),
                  ('sample', OrderedDict([
                      ('sampler', 'flowmatch'),  # must match train.noise_scheduler
                      ('sample_every', valid_steps),  # sample every this many steps
                      ('width', 1024),
                      ('height', 1024),
                      ('prompts', [
                      f'business headshot of [trigger] {os.environ.get("GENDER")}',
                  ]),
                      ('neg', ''),  # not used on flux
                      ('seed', 42),
                      ('walk_seed', True),
                      ('guidance_scale', 4),
                      ('sample_steps', 20)
                  ]))
              ])
          ])
      ])),
      ('meta', OrderedDict([
          ('name', '[name]'),
          ('version', '1.0')
      ]))
  ])
  return job_config
