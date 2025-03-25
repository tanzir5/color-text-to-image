import os
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
import numpy as np

# ------------------------- #
#      Model Definition     #
# ------------------------- #
class CLIPTextToPixel(nn.Module):
    def __init__(self, text_encoder: CLIPTextModel):
        super().__init__()
        self.text_encoder = text_encoder
        # Linear layer: maps text embeddings to 3 outputs (RGB)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)
    
    def forward(self, input_ids):
        # Obtain text embeddings using CLIP's text encoder
        outputs = self.text_encoder(input_ids=input_ids)
        # Pool token embeddings by taking the mean along the sequence dimension
        pooled = outputs.last_hidden_state.mean(dim=1)  # shape: (batch, hidden_size)
        # Predict RGB values
        pixel = self.fc(pooled)  # shape: (batch, 3)
        return pixel

# ------------------------- #
#    Inference Functions    #
# ------------------------- #
def infer_color(prompt, tokenizer, model, device, max_length=50):
    """
    Infers the color from a text prompt using the fine-tuned CLIP model.
    
    Args:
        prompt (str): The text prompt (e.g., "blue pixel").
        tokenizer: CLIPTokenizer instance.
        model: CLIPTextToPixel model.
        device (str): "cuda" or "cpu".
        max_length (int): Maximum token length.
    
    Returns:
        numpy.ndarray: Predicted RGB values as an array of integers in [0,255].
    """
    model.eval()
    tokenized = tokenizer(prompt,
                          padding="max_length",
                          truncation=True,
                          max_length=max_length,
                          return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    
    with torch.no_grad():
        pred_pixel = model(input_ids)  # shape: (batch, 3)
    
    # Remove batch dimension and convert to numpy array
    pred_pixel = pred_pixel.squeeze(0).cpu().numpy()
    
    # Ensure outputs are in [0,1] and convert to [0,255]
    rgb = (pred_pixel.clip(0, 1) * 255).astype(int)
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

# ------------------------- #
#          Main Code        #
# ------------------------- #
def main():
    # Define model name and device
    model_name = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and text encoder from Hugging Face
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name)
    
    # Instantiate the CLIPTextToPixel model and load fine-tuned weights
    model = CLIPTextToPixel(text_encoder).to(device)
    model_path = os.path.join("finetuned_model", "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights not found at {model_path}.")
        return
    
    # Example prompt; you can change this prompt as needed
    prompt = "blue pixel"
    rgb = infer_color(prompt, tokenizer, model, device)
    print(f"Predicted RGB for prompt '{prompt}': {rgb}")
    
    # Visualize the predicted color swatch
    visualize_rgb(rgb)

if __name__ == "__main__":
    main()

