import argparse
import sys
print(sys.executable)
print(sys.version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from skimage import color
from sklearn.metrics import r2_score
from transformers import RobertaTokenizer, RobertaModel
import copy
from math import inf
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
EARLY_STOP = 5

# ------------------------- #
#      Dataset Class       #
# ------------------------- #
class SinglePixelDataset(Dataset):
    """
    Reads a CSV file with color names and their RGB values.
    The CSV should have columns: 
       'Color Name', 'R', 'G', 'B'.
    Tokenizes the 'Color Name' using RoBERTa, returning
    input_ids, attention_mask, and the RGB target (normalized to [0,1]).
    """
    def __init__(self, csv_path, tokenizer, max_length=50):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the text prompt (color name)
        prompt = self.df.iloc[idx]['Color Name']
        
        # Get RGB values and normalize them to [0, 1]
        r = self.df.iloc[idx]['R']
        g = self.df.iloc[idx]['G']
        b = self.df.iloc[idx]['B']
        target_pixel = torch.tensor([r, g, b], dtype=torch.float32) / 255.0

        # Tokenize the prompt using RoBERTa’s tokenizer
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids.squeeze(0)         # shape: (max_length,)
        attention_mask = tokenized.attention_mask.squeeze(0)  # shape: (max_length,)

        return input_ids, attention_mask, target_pixel

# ------------------------- #
#   RoBERTa + CNN Head     #
# ------------------------- #
class RoBERTaTextToPixelCNN(nn.Module):
    """
    Uses RoBERTa as a text encoder and adds a CNN-based head 
    (1D convolution) to map from sequence outputs to final RGB predictions.
    """
    def __init__(self, text_encoder: RobertaModel):
        super().__init__()
        self.text_encoder = text_encoder

        hidden_size = self.text_encoder.config.hidden_size  # e.g. 768 for roberta-base
        
        # 1D CNN blocks similar to the CLIP example
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size, 
            out_channels=128, 
            kernel_size=1
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Pass inputs through RoBERTa; shape of last_hidden_state: (batch, seq_len, hidden_size)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Permute to (batch, hidden_size, seq_len) for Conv1D
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # 1x1 convolution + tanh
        x = torch.tanh(self.conv1d(hidden_states))  # (batch, 128, seq_len)
        
        # Mask padding positions before pooling
        if attention_mask is None:
            # Create default mask assuming no padding if not provided
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        # Expand mask to match dimensions (batch, 1, seq_len)
        # attention_mask[:, 0] = 0
        mask = attention_mask.unsqueeze(1)
        # Create boolean mask where padding tokens are True
        inverse_mask = (mask == 0)
        # Set padding positions to -inf to ignore them in max pooling
        x = x.masked_fill(inverse_mask, float('-inf'))

        # Global max pooling across the sequence dimension
        x = self.pool(x)  # -> (batch, 128, 1)
        x = x.squeeze(-1) # -> (batch, 128)

        x = self.dropout(x)
        x = self.fc(x)       # -> (batch, 3)
        x = self.sigmoid(x)  # -> (batch, 3) in [0,1]

        return x

# ------------------------- #
#       Training Loop       #
# ------------------------- #
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for input_ids, attention_mask, target_pixel in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_pixel = target_pixel.to(device)

        optimizer.zero_grad()
        pred_pixel = model(input_ids, attention_mask)
        loss = criterion(pred_pixel, target_pixel)
        logging.info(f'TBL={loss.item():.3f}')
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for input_ids, attention_mask, target_pixel in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets.append(target_pixel.cpu())
            predictions.append(model(input_ids, attention_mask).cpu())

    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()          # shape: (N, 3)

    # Convert to [0,255]
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
#       Metrics Logic       #
# ------------------------- #
def compute_metrics(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for input_ids, attention_mask, target in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu())
            targets.append(target.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()          # shape: (N, 3)

    # Scale from [0,1] to [0,255]
    predictions = predictions.clip(0, 1) * 255.0
    targets = targets.clip(0, 1) * 255.0

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    r2 = r2_score(targets / 255.0, predictions / 255.0)

    return rmse, mae, ciede2000, r2

# ------------------------- #
#   Param Group Creation    #
# ------------------------- #
def create_param_groups(model, base_lr=1e-4, layer_decay=0.9):
    """
    Create parameter groups with layer-wise decayed learning rates
    for the RoBERTa encoder. The 'layer_decay' indicates how much 
    we reduce the LR each time we go 1 layer 'deeper'.
    """
    param_groups = []

    # 1) CNN head parameters (gets the highest LR, i.e. base_lr)
    classifier_params = list(model.conv1d.parameters()) + list(model.fc.parameters())
    param_groups.append({
        "params": classifier_params,
        "lr": base_lr
    })

    # 2) RoBERTa encoder parameters:
    #    The main transformer blocks live in model.text_encoder.encoder.layer
    encoder_layers = list(model.text_encoder.encoder.layer)
    num_layers = len(encoder_layers)

    # We'll assign progressively decaying LRs from top to bottom
    # Let's also apply a factor (e.g. 0.01) to the base LR for the encoder
    encoder_base_lr = base_lr * 0.5

    # Start with the top-most layer (index = num_layers-1) => smaller decay => higher LR
    for layer_idx, layer in enumerate(reversed(encoder_layers)):
        layer_lr = encoder_base_lr * (layer_decay ** (layer_idx + 1))
        layer_params = list(layer.parameters())
        param_groups.append({
            "params": layer_params,
            "lr": layer_lr
        })

    # 3) Embeddings: can freeze or apply further decay
    for param in model.text_encoder.embeddings.parameters():
        param.requires_grad = True  # or False if you prefer freezing
    param_groups.append({
        "params": model.text_encoder.embeddings.parameters(),
        "lr": base_lr * (layer_decay ** (num_layers + 1))
    })

    return param_groups

# ------------------------- #
#   K-Fold Cross Validation #
# ------------------------- #
def cross_validate_roberta_model(
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
    Perform K-fold cross-validation on the RoBERTaTextToPixelCNN model.
    
    Args:
        csv_path (str): Path to the CSV dataset.
        model_name (str): Hugging Face model name (e.g., "roberta-base").
        tokenizer: The RobertaTokenizer instance.
        device (str): "cuda" or "cpu".
        lr (float): Base learning rate.
        freeze (bool): Whether to freeze encoder params.
        folds (int): Number of K-fold splits.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train per fold.
        output_dir (str): Directory to save per-fold models/results.
        decay (float): Layer-wise decay factor (1 => no decay).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the full dataset
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

        # Instantiate a fresh RoBERTa model for this fold
        text_encoder = RobertaModel.from_pretrained(model_name)
        model = RoBERTaTextToPixelCNN(text_encoder).to(device)

        # Apply freeze or layer-wise decay if specified
        if freeze:
            # Freeze the entire text encoder
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif decay != 1:
            param_groups = create_param_groups(model, base_lr=lr, layer_decay=decay)
            optimizer = optim.AdamW(param_groups)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        best_model = None
        best_val_metric = inf
        no_improve_count = 0

        # Train for num_epochs, with early stopping
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metric = validate_one_epoch(model, val_loader, device)

            if val_metric < best_val_metric:
                best_model = copy.deepcopy(model)
                best_val_metric = val_metric
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val metric (CIEDE2000): {val_metric:.4f}")

            if no_improve_count >= EARLY_STOP:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Evaluate best model
        rmse, mae, ciede2000, r2 = compute_metrics(best_model, val_loader, device)
        print(f"Fold {fold+1} Metrics: RMSE = {rmse:.2f}, MAE = {mae:.2f}, CIEDE2000 = {ciede2000:.2f}, R² = {r2:.3f}")
        fold_metrics.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'ciede2000': ciede2000,
            'r2': r2
        })

        # Save the model for this fold
        model_path = os.path.join(output_dir, f"model_fold_{fold + 1}.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Summarize metrics
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

# ------------------------- #
#         Main Script       #
# ------------------------- #
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
        help="Freezes the encoder layers"
    )
    parser.add_argument(
        "--decay", 
        type=float,
        default=1,
        help="Layer-wise Decay Rate (default is 1 => no decay)"
    )
    args = parser.parse_args()

    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = "global_training_set.csv"

    # Perform cross-validation
    cross_validate_roberta_model(
        csv_path=csv_path,
        model_name=model_name,
        tokenizer=tokenizer,
        device=device,
        folds=10,
        batch_size=160,
        num_epochs=1000,
        lr=args.lr,
        output_dir="final_roberta_cnn/",
        freeze=args.freeze,
        decay=args.decay
    )

