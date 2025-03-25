import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score  # <-- Added for R² computation
from skimage import color  # For CIEDE2000 computation

# ------------------------- #
#      Dataset Class       #
# ------------------------- #
class ColorDataset(Dataset):
    """
    Reads a CSV file with color names and their RGB values.
    The CSV should have columns: 'Color Name', 'R', 'G', 'B'.
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
        target = torch.tensor([r, g, b], dtype=torch.float32) / 255.0

        # Tokenize the prompt using RoBERTa’s tokenizer
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove the batch dimension (tokenizer returns shape [1, max_length])
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        
        return input_ids, attention_mask, target

# ------------------------- #
#      Model Definition    #
# ------------------------- #
class RobertaTextToPixel(nn.Module):
    """
    Uses RoBERTa as a text encoder and adds a linear layer to map the
    pooled text representation (from the first token) to 3 RGB values.
    """
    def __init__(self, text_encoder: RobertaModel):
        super().__init__()
        self.text_encoder = text_encoder
        # Linear layer: from hidden size (e.g., 768 for roberta-base) to 3 (RGB)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask=None):
        # Pass input through RoBERTa. We use the first token's embedding as a summary.
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
        pixel = self.fc(pooled)  # shape: (batch, 3)
        return pixel

# ------------------------- #
#  Training & Validation   #
# ------------------------- #
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for input_ids, attention_mask, target in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input_ids.size(0)
    
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for input_ids, attention_mask, target in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, target)
            total_loss += loss.item() * input_ids.size(0)
    
    return total_loss / len(dataloader.dataset)

# ------------------------- #
#  CIEDE2000 Computation    #
# ------------------------- #
def compute_ciede2000(y_true, y_pred):
    """
    Compute CIEDE2000 color difference between predicted and target colors.
    Both inputs are assumed to be in [0, 255] range.
    """
    y_true = np.array(y_true) / 255.0
    y_pred = np.array(y_pred) / 255.0
    y_true_lab = color.rgb2lab(y_true.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred.reshape(-1, 1, 3)).reshape(-1, 3)
    return color.deltaE_ciede2000(y_true_lab, y_pred_lab)

# ------------------------- #
#    Metrics Computation    #
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
    targets = torch.cat(targets, dim=0).numpy()            # shape: (N, 3)
    
    # Convert from [0,1] to [0,255]
    predictions = predictions.clip(0, 1) * 255.0
    targets = targets.clip(0, 1) * 255.0
    
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    r2 = r2_score(targets/255.0, predictions/255.0)  # Compute R² using scikit-learn's r2_score
    return rmse, mae, ciede2000, r2
# ------------------------- #
#   Cross-Validation Code  #
# ------------------------- #
def cross_validate_roberta(csv_path, folds=5, num_epochs=10, batch_size=16, lr=1e-5, output_dir="cv_results_roberta"):
    """
    Performs K-fold cross-validation for the RoBERTa-based text-to-RGB model.
    All layers remain unfrozen so that the entire model is fine-tuned.
    Now prints RMSE, MAE, CIEDE2000, and R² for each fold.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "roberta-base"
    
    # Load the tokenizer once for all folds.
    tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
    
    # Load the full dataset.
    full_dataset = ColorDataset(csv_path, tokenizer, max_length=50)
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold+1}/{folds} ===")
        
        # Create training and validation subsets.
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Instantiate a fresh RoBERTa model for this fold.
        text_encoder = RobertaModel.from_pretrained(base_model_name)
        model = RobertaTextToPixel(text_encoder).to(device)
        
        # Ensure all layers are trainable (unfrozen).
        for param in model.text_encoder.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, device)
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
            
            # Save best model for this fold.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                fold_model_path = os.path.join(output_dir, f"model_fold_{fold+1}.pt")
                torch.save(model.state_dict(), fold_model_path)
                print(f"  Best model for Fold {fold+1} saved at epoch {epoch+1}")
        
        # Compute evaluation metrics on the validation set.
        rmse, mae, ciede2000, r2 = compute_metrics(model, val_loader, device)
        print(f"Fold {fold+1} Metrics: RMSE = {rmse:.2f}, MAE = {mae:.2f}, CIEDE2000 = {ciede2000:.2f}, R² = {r2:.3f}")
        fold_metrics.append({
            'fold': fold+1,
            'rmse': rmse,
            'mae': mae,
            'ciede2000': ciede2000,
            'r2': r2
        })
    
    # Save overall metrics.
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
#         Main Script      #
# ------------------------- #
if __name__ == "__main__":
    # Replace with the path to your CSV file containing your training data.
    csv_path = "global_training_set.csv"
    cross_validate_roberta(csv_path, folds=10, num_epochs=10, batch_size=16, lr=1e-5)

