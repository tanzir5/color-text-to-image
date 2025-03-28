import argparse
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



class CLIPTextToPixelCNN(nn.Module):
    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder
        hidden_size = self.text_encoder.config.hidden_size  # e.g. 768 for "openai/clip-vit-large-patch14"

        # Conv1D in PyTorch uses (batch, in_channels, seq_len),
        # so we'll permute the encoder outputs.
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        # text_encoder outputs [batch_size, seq_len, hidden_size]
        outputs = self.text_encoder(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Permute to (batch, hidden_size, seq_len) for Conv1D
        hidden_states = hidden_states.permute(0, 2, 1)  # now (batch, in_channels=hidden_size, seq_len)

        # 1x1 Conv (pointwise) + tanh
        x = torch.tanh(self.conv1d(hidden_states))

        # Global max pooling across the sequence dimension
        x = self.pool(x)  # -> (batch, out_channels=128, 1)
        x = x.squeeze(-1)  # -> (batch, 128)

        x = self.dropout(x)
        x = self.fc(x)        # -> (batch, 3)
        x = self.sigmoid(x)   # -> (batch, 3) in [0,1]

        return x


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
    total_ciede = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for input_ids, target_pixel in dataloader:
            input_ids = input_ids.to(device)
            targets.append(target_pixel.cpu())
            predictions.append(model(input_ids).cpu())
    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()            # shape: (N, 3)
    predictions *= 255.0
    targets *= 255.0
    return compute_ciede2000(targets, predictions).mean() 

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


def create_param_groups(model, base_lr=1e-4, layer_decay=0.9):
    """
    Create parameter groups with layer-wise decayed learning rates.
    The 'layer_decay' indicates how much we reduce the LR each time we go 1 layer 'deeper'.
    """
    param_groups = []

    # 1) Classifier/head parameters (gets the highest LR, i.e. base_lr)
    classifier_params = list(model.conv1d.parameters()) + \
                        list(model.fc.parameters())
    param_groups.append({
        "params": classifier_params,
        "lr": base_lr
    })

    # 2) Text encoder parameters: we retrieve them from the last layer to the first
    #    and assign progressively decaying LRs.
    #    In CLIPTextModel, the main layers often live in model.text_encoder.encoder.layers
    encoder_layers = list(model.text_encoder.text_model.encoder.layers)

    num_layers = len(encoder_layers)
    # e.g. in CLIP, if you want to also include final layers like layer_norm, etc.,
    # you can adapt as needed or gather them separately.
    
    # Start with the last layer (closest to output) => smaller decay => higher LR
    # Move to the first layer => bigger decay => smaller LR
    encoder_base_lr = base_lr * 0.01
    for layer_idx, layer in enumerate(reversed(encoder_layers)):
        # Compute the decayed LR for this layer
        # layer_idx = 0 means the top-most layer (closest to output)
        layer_lr = encoder_base_lr * (layer_decay ** (layer_idx + 1))

        # Collect layer’s parameters
        layer_params = list(layer.parameters())
        param_groups.append({
            "params": layer_params,
            "lr": layer_lr
        })

    # 3) Optionally: If you want an even smaller LR for the embedding layer,
    #    freeze it or apply further decay:
    for param in model.text_encoder.text_model.embeddings.parameters():
        param.requires_grad = True  # or False if you want it frozen
    param_groups.append({
        "params": model.text_encoder.text_model.embeddings.parameters(),
        "lr": base_lr * (layer_decay ** (num_layers + 1))
    })

    return param_groups


# ------------------------- #
#   K-Fold Cross Validation #
# ------------------------- #
def cross_validate_clip_model(
    csv_path, 
    model_name, 
    tokenizer, 
    device, 
    lr,
    freeze, 
    folds, 
    batch_size, 
    num_epochs, 
    output_dir,
    decay
):
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
        model = CLIPTextToPixelCNN(text_encoder_fold).to(device)
        
        if freeze:
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif decay != 1:
            param_groups = create_param_groups(model, base_lr=lr, layer_decay=decay)
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_model= None
        best_val_metric = inf
        no_improve_count = 0
        
        # Train for a fixed number of epochs
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metric = validate_one_epoch(model, val_loader, device)
            if val_metric < best_val_metric:
                best_model = copy.deepcopy(model)
                best_val_metric = val_metric
                no_improve_count = 0
            else:
                no_improve_count +=1
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val metric: {val_metric:.4f}")
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
        torch.save(best_model.state_dict(), model_path)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", 
        type=float,
        default=1e-5,
        help="Learning Rate (default is 1e-5)"
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freezes the embedding layers"
    )
    parser.add_argument(
        "--decay", 
        type=float,
        default=1,
        help="Decay Rate (default is 1 which means no decay)"
    )
    args = parser.parse_args()

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
        lr=args.lr,
        output_dir="final_clip/",
        freeze=args.freeze,
        decay=args.decay
    )

    
