import os
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import copy
from transformers import RobertaTokenizer, RobertaModel

from src.finetune_clip import CLIPTextToPixelCNN
from src.finetune_roberta import RoBERTaTextToPixelCNN

CONCEPT = 'color_name'

# ------------------------- #
#    Inference Functions    #
# ------------------------- #
def run_clip(prompt, tokenizer, model, device, max_length=50):
    """
    Infers the color from a text prompt using the fine-tuned CLIP/roBERTa model.
    
    Args:
        prompt (str): The text prompt (e.g., "blue pixel").
        tokenizer: CLIPTokenizer instance.
        model: CLIPTextToPixel model.
        device (str): "cuda" or "cpu".
        max_length (int): Maximum token length.
    
    Returns:
        numpy.ndarray: Predicted RGB values as an array of integers in [0,255].
    """
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = tokenized.input_ids.to(device)
    
    with torch.no_grad():
        pred_pixel = model(input_ids)  # shape: (batch, 3)
    
    # Remove batch dimension and convert to numpy array
    print(pred_pixel.shape)
    pred_pixel = pred_pixel.squeeze(0).cpu().numpy()
    print(pred_pixel.shape)
    # Ensure outputs are in [0,1] and convert to [0,255]
    rgb = (pred_pixel.clip(0, 1) * 255).astype(float)
    return rgb

def run_roberta(prompts, tokenizer, model, device, max_length=50):
    tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    with torch.no_grad():
        pred_pixel = model(input_ids, attention_mask)
    
    pred_pixel = pred_pixel.squeeze(0).cpu().numpy()
 
    rgb = (pred_pixel.clip(0, 1) * 255).astype(float)
    return rgb

def visualize_rgb(rgb, swatch_size=100):
    """
    Visualizes an RGB color swatch using Matplotlib.
    
    Args:
        rgb (array-like): RGB values (list/array of 3 integers in 0–255).
        swatch_size (int): Size (in pixels) of the color swatch.
    """
    # Normalize RGB values to [0,1]
    normalized_rgb = np.array(rgb) / 255.0
    # Create a swatch image filled with the predicted color
    swatch = np.full((swatch_size, swatch_size, 3), normalized_rgb, dtype=np.float32)
    
    plt.figure(figsize=(2, 2))
    plt.imshow(swatch)
    plt.axis('off')
    plt.title(f"RGB: {rgb}")
    plt.show()

def roberta_infer_df(df, roberta_path):
    model_name = "roberta-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and text encoder from Hugging Face
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text_encoder = RobertaModel.from_pretrained(model_name)
    model = RoBERTaTextToPixelCNN(text_encoder).to(device)

    
    if os.path.exists(roberta_path):
        print(f"model.conv1d.weight.shape = {model.conv1d.weight.shape}")
        print(f"before loading, model embedding weight: {model.conv1d.weight[1, 0].detach()}")
        model.load_state_dict(torch.load(roberta_path, map_location=device))
        print(f"after loading, model embedding weight: {model.conv1d.weight[1, 0].detach()}")
    else:
        print(f"Model weights not found at {roberta_path}.")
        return
    model.eval()
    # Example prompt; you can change this prompt as needed
    
    prompts = df[CONCEPT].tolist()
    # prompts = 'Apple'
    rgb = run_roberta(prompts, tokenizer, model, device)
    print(rgb)
    df['r_roberta'] = [ item[0] for item in rgb]
    df['g_roberta'] = [ item[1] for item in rgb]
    df['b_roberta'] = [ item[2] for item in rgb]
    return df


def clip_infer_df(df, clip_path):
    model_name = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and text encoder from Hugging Face
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name)

    # Instantiate the CLIPTextToPixel model and load fine-tuned weights
    model = CLIPTextToPixelCNN(text_encoder).to(device)
    if os.path.exists(clip_path):
        print(f"before loading, model embedding weight: {model.conv1d.weight[1, 0].detach()}")
        model.load_state_dict(torch.load(clip_path, map_location=device))
        print(f"after loading, model embedding weight: {model.conv1d.weight[1, 0].detach()}")
    else:
        print(f"Model weights not found at {clip_path}.")
        return
    model.eval()
    # Example prompt; you can change this prompt as needed

    prompts = df[CONCEPT].tolist()
    # prompts = 'aPPLe'
    rgb = run_clip(prompts, tokenizer, model, device)
    print(rgb)
    df['r_clip'] = [ item[0] for item in rgb]
    df['g_clip'] = [ item[1] for item in rgb]
    df['b_clip'] = [ item[2] for item in rgb]
    return df


# ------------------------- #
#          Main Code        #
# ------------------------- #
def main():
    # Define model name and device
    df = pd.read_csv(sys.argv[1])
    output_path = sys.argv[2]
    clip_path = sys.argv[3]
    df = clip_infer_df(df, clip_path)
    if len(sys.argv) == 5:
        roberta_path = sys.argv[4]
        df = roberta_infer_df(df, roberta_path)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()


