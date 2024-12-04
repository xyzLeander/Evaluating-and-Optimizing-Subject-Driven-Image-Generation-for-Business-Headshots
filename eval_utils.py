import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from transformers import ViTModel, CLIPProcessor, CLIPModel, ViTImageProcessor
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from identity_stability import IdentityStability
from pick_score import PickScore
from identity_preservation import IdentityPreservation
from image_reward import ImageReward
from aesthetic_predictor import AestheticPredictor
from ultralytics import YOLO
import supervision as sv
import cv2
import torchvision
import os
import random
import torchvision.models as models
from sklearn.preprocessing import MinMaxScaler


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calc_aes(gen_images):
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit([[1], [10]])  # Fit the scaler to the range of aesthetic scores

  aesthetic_predictor = AestheticPredictor(DEVICE)
  aes_results = aesthetic_predictor(gen_images)
  aes_results = aes_results.reshape(-1, 1)
  normalized_results = scaler.transform(aes_results)
  return np.mean(normalized_results)

def calc_pickscore(prompts, gen_images):
  pick_score = PickScore(DEVICE)
  pickscore_results = pick_score(gen_images, prompts)
  return pickscore_results.get('pick_score').mean()

def calc_image_reward_score(prompts,  gen_images):
  image_reward = ImageReward(DEVICE)
  imagereward_results = image_reward(gen_images, prompts)
  return imagereward_results.get('image_reward').mean()

def calc_ips(train_images, gen_images):
  IPS = IdentityPreservation(DEVICE)
  ips_results = IPS(train_images, gen_images).mean()
  return ips_results

def calc_dino(train_images, gen_images):
  #  Calculate embeddings for both sets
  generated_embeddings = calculate_dino_embeddings(gen_images)
  real_embeddings = calculate_dino_embeddings(train_images)
  dino_results = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()

  return float(dino_results)

def calc_clipi(train_images, gen_images):
  generated_embeddings = calculate_clip_embeddings(gen_images)
  real_embeddings = calculate_clip_embeddings(train_images)
  clipi_results = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()
  
  return float(clipi_results)

def load_inference_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
            img = Image.open(os.path.join(path, file))
            images.append(img)
    return images

def load_aux_images(path):
  images = []
  for file in os.listdir(path):
      if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
          img = Image.open(os.path.join(path, file))
          images.append(img)
  return images

def calc_clip_score(train_images, images):
  generated_embeddings = calc_clip_embeddings(images)
  real_embeddings = calc_clip_embeddings(train_images)
  metric = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()
  return float(metric)


def calc_sis(aux_images, gen_images):
  # pick 5 random aux images (not involved in training) and 5 random generated images, to calculate identity stability score.
  # this is done because the dataset does not include 10 additional unused images for every subject to accomplish a full identity stability evaluation.
  # Wrap aux images in a list for SIS to process them.
  random_aux_images = [[img] for img in random.sample(aux_images, min(5, len(aux_images)))]
  random_gen_images = random.sample(gen_images, min(5, len(gen_images)))

  identity_stability = IdentityStability(DEVICE)
  sis_results = identity_stability(random_aux_images, random_gen_images)

  return np.array(sis_results).mean()

def calc_yolo_subject_preservation(train_images, gen_images):
  model = YOLO("yolo11x.pt")
  cv2_train_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in train_images]
  cv2_gen_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in gen_images]
  gen_detections = []
  train_detections = []

  train_results = model(train_images)
  gen_results = model(gen_images)

  for result in train_results:

    detections = sv.Detections.from_ultralytics(result)
    detections_dict = detections.__dict__

    person_indices = [i for i, x in enumerate(detections_dict['data']['class_name']) if x == 'person']
    person_confidences = detections_dict['confidence'][person_indices]
    max_confidence_index = person_indices[np.argmax(person_confidences)]

    train_detection = {
    'xyxy': detections_dict['xyxy'][max_confidence_index].reshape(1, -1),
    'confidence': detections_dict['confidence'][max_confidence_index].reshape(1),
    'class_id': detections_dict['class_id'][max_confidence_index].reshape(1),
    'data': {'class_name': [detections_dict['data']['class_name'][max_confidence_index]]}
    }
    train_detections.append(train_detection)

  for result in gen_results:

    detections = sv.Detections.from_ultralytics(result)
    detections_dict = detections.__dict__

    person_indices = [i for i, x in enumerate(detections_dict['data']['class_name']) if x == 'person']
    person_confidences = detections_dict['confidence'][person_indices]
    max_confidence_index = person_indices[np.argmax(person_confidences)]

    gen_detection = {
    'xyxy': detections_dict['xyxy'][max_confidence_index].reshape(1, -1),
    'confidence': detections_dict['confidence'][max_confidence_index].reshape(1),
    'class_id': detections_dict['class_id'][max_confidence_index].reshape(1),
    'data': {'class_name': [detections_dict['data']['class_name'][max_confidence_index]]}
    }
    gen_detections.append(gen_detection)

  # print(cv2_train_images)
  train_embeddings = subject_preservation_embeddings(cv2_train_images, train_detections)
  gen_embeddings = subject_preservation_embeddings(cv2_gen_images, gen_detections)

  metric = F.cosine_similarity(gen_embeddings, train_embeddings, dim=1).mean()
  return float(metric)

def subject_preservation_embeddings(cv2_image_list, detections_list):
  # Define the feature extraction model (e.g., a CNN)
  resnet = torchvision.models.resnet50(pretrained=True)
  resnet.eval()  # Set the model to evaluation mode

  # Define the image transform to preprocess the cropped images
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize to a fixed size (e.g., 224x224)
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  for i, (image, detections) in enumerate(zip(cv2_image_list, detections_list)):
      # Extract the bounding box coordinates
      xyxy = detections_list[i]['xyxy']

      x0 = xyxy[0][0] 
      y0 = xyxy[0][1]
      x1 = xyxy[0][2]
      y1 = xyxy[0][3]


      # Crop the original image using the bounding box coordinates
      crop = image[int(y0):int(y1), int(x0):int(x1)]

      # Convert the crop to a PIL image
      crop_img = Image.fromarray(crop)

      # Preprocess the cropped image
      input_img = transform(crop_img)

      # Extract the embeddings
      with torch.no_grad():
          embeddings = resnet(input_img.unsqueeze(0))  # Add a batch dimension

  return embeddings


# Function to calculate embeddings
def calculate_dino_embeddings(images):
    # Define transformations
    transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Define model
    model = ViTModel.from_pretrained('facebook/dino-vits8')

    with torch.no_grad():
        inputs = []
        for image in images:
            # Ensure the image is RGB and has 3 channels
            if image.mode!= 'RGB':
                image = image.convert('RGB')
            inputs.append(transform(image))

        inputs = torch.stack(inputs)
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state  # ViT backbone features

    return last_hidden_states[:, 0]  # Get cls token (0-th token) for each image

def calc_clip_embeddings(images):
  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

  inputs = processor(images=images, return_tensors="pt")
  image_features = clip_model.get_image_features(inputs["pixel_values"])
  return image_features


def calculate_clip_embeddings(images):
  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

  inputs = processor(images=images, return_tensors="pt")
  image_features = clip_model.get_image_features(inputs["pixel_values"])
  return image_features

def calc_dino_score(training_images, generated_images):
  # Calculate embeddings for both sets
  generated_embeddings = calc_dino_embeddings(generated_images)
  real_embeddings = calc_dino_embeddings(training_images)
  metric = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()
  return float(metric)

# Function to calculate embeddings
def calc_dino_embeddings(images):
    # Define transformations
    transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Define model
    model = ViTModel.from_pretrained('facebook/dino-vits8')

    with torch.no_grad():
        inputs = torch.stack([transform(image) for image in images])
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state  # ViT backbone features
        
    return last_hidden_states[:, 0]  # Get cls token (0-th token) for each img

# dreambooth dino model, results seem the same with this model however.
def calc_dino2_score(training_images, generated_images):
  # Calculate embeddings for both sets
  generated_embeddings = calc_dino_embeddings(generated_images)
  real_embeddings = calc_dino_embeddings(training_images)
  metric = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()
  return float(metric)

# Function to calculate embeddings
def calc_dino2_embeddings(images):
    # Define transformations
    # Define model
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16')
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state  # ViT backbone features
        
    return last_hidden_states[:, 0]  # Get cls token (0-th token) for each img

def calc_haar_subject_preservation(real_images, gen_images):
  generated_embeddings, gen_detection_info = calculate_haar_embeddings(gen_images)
  real_embeddings, real_detection_info = calculate_haar_embeddings(real_images)

  # Ensure the concatenated tensors have the same shape by truncating the longer tensor
  min_length = min(generated_embeddings.shape[0], real_embeddings.shape[0])
  generated_embeddings = generated_embeddings[:min_length]
  real_embeddings = real_embeddings[:min_length]

  metric = F.cosine_similarity(generated_embeddings, real_embeddings, dim=1).mean()
  return float(metric), real_detection_info, gen_detection_info


def calculate_haar_embeddings(images):
  # Image preprocessing (modify as needed)
  transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  ])

  # Load pre-trained models (replace with your choices)
  model = models.resnet50(pretrained=True)
  model.eval()
  # body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
  body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
  upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

  bodies = []
  upper_bodies = []
  faces = []
  cropped_images = []
  image_features = []
  images_with_no_body = []
  detection_info = []


  for i, image in enumerate(images):
      # Convert the image to a NumPy array
      image_array = np.array(image)

      # Detect bodies in the image
      body = body_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
      upper_body = upper_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
      face = face_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

      # Check if any bodies were detected
      if body is not None and len(body) > 0:
          x, y, w, h = body[0]
          cropped_image = transform(image.crop((x, y, x + w, y + h))).unsqueeze(0)
          cropped_images.append(cropped_image)
          bodies.append(body[0])
          detection_info.append({"image_index": i, "detection": "body"})
      elif upper_body is not None and len(upper_body) > 0:
          x, y, w, h = upper_body[0]
          cropped_image = transform(image.crop((x, y, x + w, y + h))).unsqueeze(0)
          cropped_images.append(cropped_image)
          upper_bodies.append(upper_body[0])
          detection_info.append({"image_index": i, "detection": "upper body"})
      elif face is not None and len(face) > 0:
          x, y, w, h = face[0]
          cropped_image = transform(image.crop((x, y, x + w, y + h))).unsqueeze(0)
          cropped_images.append(cropped_image)
          faces.append(face[0])
          detection_info.append({"image_index": i, "detection": "face"})
      else:    
          images_with_no_body.append(image)
          detection_info.append({"image_index": i, "detection": "nothing"})

  # Process the cropped images to get features
  for cropped_image in cropped_images:
        with torch.no_grad():
            image_features.append(model(cropped_image))

  # Concatenate the list of tensors into a single tensor along the batch dimension (dim=0)
  if image_features:
      image_features = torch.cat(image_features, dim=0)
  else:
      image_features = torch.tensor([])  # Return an empty tensor if no features are found

  return image_features, detection_info



# class CLIP:
#     def __init__(
#         self,
#         device,
#         dtype,
#         model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
#     ) -> None:
#         self.dtype = dtype
#         self.device = device
#         self.model: CLIPModel = CLIPModel.from_pretrained(model_name).to(
#             device=device, dtype=dtype
#         )
#         self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
#         self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
#         self.model.eval()

#     def prepare_images(self, images: list[Image.Image]):
#         return torch.stack([self.image_transforms()(image) for image in images]).to(
#             dtype=self.dtype, device=self.device
#         )

#     @torch.inference_mode()
#     def __call__(
#         self,
#         images: list[Image.Image],
#         prompt: list[str] | None,
#     ):
#         inputs = {}
#         if prompt is None:
#             prompt = [""] * len(images)

#         inputs = self.tokenizer(
#             prompt,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#         ).to(self.device)
#         pixel_values = torch.from_numpy(
#             np.stack(self.image_processor(images)["pixel_values"])
#         ).to(self.device)
#         outputs = self.model(pixel_values=pixel_values, **inputs)

#         return {
#             "text_embeds": outputs.text_embeds,
#             "image_embeds": outputs.image_embeds,
#         }