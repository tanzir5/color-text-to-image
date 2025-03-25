import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score  # For R² computation
from skimage import color         # For CIEDE2000 computation
import gensim

# ------------------------- #
#   Data Preparation Code   #
# ------------------------- #
def read_colors_custom_dataset():
    """
    Read and format color dataset.
    """
    colors = pd.read_csv('global_training_set.csv', encoding='ISO-8859-1')
    colors = colors[colors['Color Name'].notna()]
    return {row['Color Name']: {'r': row['R'], 'g': row['G'], 'b': row['B']} for _, row in colors.iterrows()}

def prepare_data(rgbs, word2vec):
    """
    Prepare input (X) and output (Y) data.
    """
    max_tokens = max(len(name.split()) for name in rgbs)
    dim = 300  # Embedding dimension
    avg_vec = np.mean(word2vec.vectors, axis=0)
    empty_vec = np.zeros(dim)

    X, Y, color_names = [], [], []
    for name, rgb in rgbs.items():
        tokens = name.lower().split()
        # For each token, if it's in the word2vec vocabulary use its embedding,
        # otherwise use the average vector; pad with empty vectors if needed.
        token_vecs = [word2vec[token] if token in word2vec else avg_vec for token in tokens]
        # Padding to match max_tokens
        padding = [empty_vec] * (max_tokens - len(tokens))
        X.append(token_vecs + padding)
        Y.append([rgb[c] for c in ['r', 'g', 'b']])
        color_names.append(name)

    return np.array(X), np.array(Y) / 255.0, color_names

# ------------------------- #
#      Dataset Class       #
# ------------------------- #
class Word2VecDataset(Dataset):
    """
    Dataset for word2vec-based color data.
    X: numpy array of shape (num_samples, max_tokens, 300)
    Y: numpy array of shape (num_samples, 3)
    """
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ------------------------- #
#      Model Definition    #
# ------------------------- #
class Word2VecToPixel(nn.Module):
    """
    Maps a sequence of word2vec embeddings to an RGB value.
    The input is a tensor of shape (batch, max_tokens, 300). Here we simply average
    the embeddings over tokens and then apply a linear layer to predict the RGB values.
    """
    def __init__(self, input_dim=300):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)

    def forward(self, x):
        # x shape: (batch, max_tokens, 300)
        # Compute the mean embedding over tokens
        x_avg = x.mean(dim=1)  # shape: (batch, 300)
        rgb = self.fc(x_avg)   # shape: (batch, 3)
        return rgb

# ------------------------- #
#  Training & Validation   #
# ------------------------- #
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    
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
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu())
            targets.append(y_batch.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()            # shape: (N, 3)
    
    # Convert from [0,1] to [0,255]
    predictions = np.clip(predictions, 0, 1) * 255.0
    targets = np.clip(targets, 0, 1) * 255.0
    
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    r2 = r2_score(targets, predictions)
    
    return rmse, mae, ciede2000, r2

# ------------------------- #
#   Cross-Validation Code  #
# ------------------------- #
def cross_validate_word2vec(folds=5, num_epochs=10, batch_size=16, lr=1e-5, output_dir="cv_results_word2vec"):
    """
    Performs K-fold cross-validation for the word2vec-based model.
    It reads the global color dataset, prepares the data using word2vec embeddings,
    and then trains the model mapping the averaged embedding to RGB values.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load color dataset and word2vec embeddings.
    rgbs = read_colors_custom_dataset()
    # Load a pre-trained word2vec model (for example, GoogleNews-vectors-negative300).
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-001.bin', binary=True)
    
    # Prepare data using the provided helper function.
    X, Y, color_names = prepare_data(rgbs, word2vec)
    full_dataset = Word2VecDataset(X, Y)
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
        
        # Instantiate a fresh model for this fold.
        model = Word2VecToPixel(input_dim=300).to(device)
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
    # Make sure that the file paths (for CSV and word2vec model) are correct.
    cross_validate_word2vec(folds=10, num_epochs=10, batch_size=16, lr=1e-5)

