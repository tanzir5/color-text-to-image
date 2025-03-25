import sys
print(sys.executable)
print(sys.version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
import torchvision.transforms as transforms
from sklearn.metrics import r2_score
import os
import numpy as np
from sklearn.model_selection import KFold
from skimage import color
from tqdm import tqdm
import logging
import copy
from math import inf

logging.basicConfig(level=logging.INFO)
EARLY_STOP = 5

# 1. Define a Dataset that reads CSV (first column: prompt, second column: image path)
class SinglePixelDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=50):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Transformation to convert any image into a single pixel.
        # We load the image, convert to RGB, then resize to 1x1.
        self.transform = transforms.Compose([
            transforms.Resize((1, 1)),
            transforms.ToTensor()  # converts to [C,H,W] with values in [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.df.iloc[idx]['Color Name']  # Access color name as prompt

        # Read RGB values
        r = self.df.iloc[idx]['R']
        g = self.df.iloc[idx]['G']
        b = self.df.iloc[idx]['B']

        # Normalize RGB to [0, 1]
        target_pixel = torch.tensor([r, g, b], dtype=torch.float) / 255.0

        # Tokenize the prompt using CLIP tokenizer
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids.squeeze(0)

        return input_ids, target_pixel


# 2. Define a model that uses the actual CLIP text encoder from Hugging Face
#    (This is the real model from Hugging Face, not a toy MLP.)
class CLIPTextToPixel(nn.Module):
    def __init__(self, text_encoder: CLIPTextModel):
        super().__init__()
        self.text_encoder = text_encoder
        # The CLIP text encoder outputs embeddings of dimension 768 for "openai/clip-vit-large-patch14"
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)

    def forward(self, input_ids):
        # Obtain text embeddings from the CLIP text encoder.
        outputs = self.text_encoder(input_ids=input_ids)
        # Pool the token embeddings (here we take the mean of the last hidden state)
        pooled = outputs.last_hidden_state.mean(dim=1)  # shape: (batch, hidden_size)
        # Map to 3 outputs (RGB)
        pixel = self.fc(pooled)  # shape: (batch, 3)
        return pixel


# 6. Define a simple training loop using MSE loss.
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for input_ids, target_pixel in  tqdm(dataloader):
        input_ids = input_ids.to(device)         # shape: (batch, max_length)
        target_pixel = target_pixel.to(device)     # shape: (batch, 3)

        optimizer.zero_grad()
        pred_pixel = model(input_ids)              # shape: (batch, 3)
        loss = criterion(pred_pixel, target_pixel)
        logging.info(f'TBL={loss.item():.3f}')
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for input_ids, target_pixel in dataloader:
            input_ids = input_ids.to(device)
            target_pixel = target_pixel.to(device)
            pred_pixel = model(input_ids)
            loss = criterion(pred_pixel, target_pixel)
            total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


# ------------------------- #
#  CIEDE2000 Computation    #
# ------------------------- #
def compute_ciede2000(y_true, y_pred):
    """
    Compute CIEDE2000 color difference per predicted color.
    Both inputs are assumed to be in [0, 255] range.
    """
    y_true = np.array(y_true) / 255.0
    y_pred = np.array(y_pred) / 255.0
    y_true_lab = color.rgb2lab(y_true.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred.reshape(-1, 1, 3)).reshape(-1, 3)
    return color.deltaE_ciede2000(y_true_lab, y_pred_lab)


# ------------------------- #
#     Metrics Computation   #
# ------------------------- #
def compute_metrics(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for input_ids, target in dataloader:
            input_ids = input_ids.to(device)
            target = target.to(device)
            outputs = model(input_ids)  # For CLIP, you might use just input_ids if no attention mask is needed
            predictions.append(outputs.cpu())
            targets.append(target.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()            # shape: (N, 3)

    # Scale from [0,1] to [0,255]
    predictions = predictions.clip(0, 1) * 255.0
    targets = targets.clip(0, 1) * 255.0

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    r2 = r2_score(targets/255.0, predictions/255.0)

    return rmse, mae, ciede2000, r2


# ------------------------- #
#   K-Fold Cross Validation #
# ------------------------- #
def cross_validate_clip_model(csv_path, model_name, tokenizer, device, folds=5, batch_size=16, num_epochs=10, lr=1e-4, output_dir="cv_results"):
    """
    Perform K-fold cross-validation on the CLIPTextToPixel model.
    
    Args:
        csv_path (str): Path to the CSV dataset.
        model_name (str): Hugging Face model name (e.g., "openai/clip-vit-large-patch14").
        tokenizer: The CLIPTokenizer instance.
        device (str): "cuda" or "cpu".
        folds (int): Number of folds for cross-validation.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train per fold.
        lr (float): Learning rate.
        output_dir (str): Directory to save per-fold models and results.
    
    Returns:
        metrics_df (pd.DataFrame): DataFrame of metrics for each fold.
        summary (pd.DataFrame): Summary (mean and std) of metrics across folds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load full dataset using your defined dataset class
    full_dataset = SinglePixelDataset(csv_path, tokenizer, max_length=50)
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold + 1}/{folds} ===")
        
        # Create subsets for training and validation
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Instantiate a fresh model for this fold by reloading the text encoder
        text_encoder_fold = CLIPTextModel.from_pretrained(model_name)
        model = CLIPTextToPixel(text_encoder_fold).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_model= None
        best_val_loss = inf
        no_improve_count = 0
        
        # Train for a fixed number of epochs
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, device)
            if val_loss<best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count +=1
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if no_improve_count>=EARLY_STOP:
                break
        rmse, mae, ciede2000, r2 = compute_metrics(best_model, val_loader, device)
        print(f"Fold {fold+1} Metrics: RMSE = {rmse:.2f}, MAE = {mae:.2f}, CIEDE2000 = {ciede2000:.2f}, R² = {r2:.3f}")
        fold_metrics.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'ciede2000': ciede2000,
            'r2': r2
        })
     
        # Save the model for this fold (optional)
        model_path = os.path.join(output_dir, f"model_fold_{fold + 1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.agg({
        'rmse': ['mean', 'std'], 
        'mae': ['mean', 'std'], 
        'ciede2000': ['mean', 'std'],
        'r2': ['mean', 'std']
    })
    
    metrics_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    
    print("\n=== Cross-Validation Summary ===")
    print(summary)
    
    return metrics_df, summary


# Define your model name (Hugging Face)
model_name = "openai/clip-vit-large-patch14"

# Download the tokenizer (if not already done)
tokenizer = CLIPTokenizer.from_pretrained(model_name)

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Run cross-validation on your CSV dataset
csv_path = "global_training_set.csv"  # or your modified CSV (or sampled CSV)
metrics_df, summary = cross_validate_clip_model(
    csv_path=csv_path,
    model_name=model_name,
    tokenizer=tokenizer,
    device=device,
    folds=10,          # For example, 10-fold CV
    batch_size=160,     # You can try batch sizes of 16 or 32
    num_epochs=1000,     # Adjust epochs based on training behavior
    lr=5e-6,
    output_dir="cv_results"
)

