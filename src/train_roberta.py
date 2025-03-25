import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from skimage import color  # For CIEDE2000 computation
import torchvision.transforms as transforms

# ------------------------- #
#      SinglePixelDataset    #
# ------------------------- #
class SinglePixelDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=50):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Although this transformation is defined,
        # we are directly reading the RGB values from the CSV.
        self.transform = transforms.Compose([
            transforms.Resize((1, 1)),
            transforms.ToTensor()  # Converts to [C, H, W] with values in [0, 1]
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        prompt = self.df.iloc[idx]['Color Name']  # Access color name as prompt

        # Read RGB values and normalize to [0, 1]
        r = self.df.iloc[idx]['R']
        g = self.df.iloc[idx]['G']
        b = self.df.iloc[idx]['B']
        target_pixel = torch.tensor([r, g, b], dtype=torch.float) / 255.0

        # Tokenize the prompt using the CLIP (or in our case, any) tokenizer
        tokenized = self.tokenizer(prompt,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=self.max_length,
                                   return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze(0)

        return input_ids, target_pixel

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
        # Linear layer mapping from hidden_size (e.g., 768) to 3 outputs (R, G, B)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)

    def forward(self, input_ids):
        # Get text embeddings from RoBERTa.
        outputs = self.text_encoder(input_ids=input_ids)
        # Use the embedding of the first token (acting like a CLS token) as a summary.
        pooled = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
        pixel = self.fc(pooled)  # shape: (batch, 3)
        return pixel

# ------------------------- #
#  CIEDE2000 Computation    #
# ------------------------- #
def compute_ciede2000(y_true, y_pred):
    """
    Compute CIEDE2000 color difference between predicted and target colors.
    Both inputs are assumed to be in [0, 255].
    """
    y_true = np.array(y_true) / 255.0
    y_pred = np.array(y_pred) / 255.0
    y_true_lab = color.rgb2lab(y_true.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred.reshape(-1, 1, 3)).reshape(-1, 3)
    return color.deltaE_ciede2000(y_true_lab, y_pred_lab)

# ------------------------- #
#  Training & Validation   #
# ------------------------- #
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for input_ids, target in dataloader:
        input_ids = input_ids.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
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
        for input_ids, target in dataloader:
            input_ids = input_ids.to(device)
            target = target.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, target)
            total_loss += loss.item() * input_ids.size(0)
    
    return total_loss / len(dataloader.dataset)

def compute_metrics(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for input_ids, target in dataloader:
            input_ids = input_ids.to(device)
            target = target.to(device)
            outputs = model(input_ids)
            predictions.append(outputs.cpu())
            targets.append(target.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()  # shape: (N, 3)
    targets = torch.cat(targets, dim=0).numpy()            # shape: (N, 3)
    
    # Convert predictions and targets from [0,1] to [0,255]
    predictions = predictions.clip(0, 1) * 255.0
    targets = targets.clip(0, 1) * 255.0
    
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mae = np.mean(np.abs(predictions - targets))
    ciede2000 = compute_ciede2000(targets, predictions).mean()
    
    return rmse, mae, ciede2000

# ------------------------- #
#      Final Training      #
# ------------------------- #
def train_final_roberta_model(csv_path, num_epochs=50, batch_size=32, lr=1e-5, val_split=0.2, patience=3, output_dir="final_model_roberta"):
    """
    Train the final RoBERTa-based text-to-RGB model on the entire global training dataset.
    Uses a hold-out validation split and early stopping to avoid overfitting.
    After training, the final metrics (RMSE, MAE, CIEDE2000) are saved to a CSV file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "roberta-base"
    
    # Load tokenizer and pre-trained text encoder for RoBERTa.
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text_encoder = RobertaModel.from_pretrained(model_name)
    
    # Instantiate the model.
    model = RobertaTextToPixel(text_encoder).to(device)
    # Ensure all layers are trainable (they are unfrozen by default).
    for param in model.text_encoder.parameters():
        param.requires_grad = True
    
    # Load the global dataset.
    dataset = SinglePixelDataset(csv_path, tokenizer, max_length=50)
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    print(f"Starting training on {total_samples} samples (Train: {train_size}, Val: {val_size})...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Early stopping: Save best model if validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model_save_path = os.path.join(output_dir, "model.pt")
            torch.save(model.state_dict(), model_save_path)
            print("  Model improved and saved!")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. Training halted.")
                break
    
    # Compute final metrics on the validation set.
    rmse, mae, ciede2000 = compute_metrics(model, val_loader, device)
    print(f"Final Validation Metrics: RMSE = {rmse:.2f}, MAE = {mae:.2f}, CIEDE2000 = {ciede2000:.2f}")
    
    # Save the metrics to a CSV file.
    metrics = {"rmse": [rmse], "mae": [mae], "ciede2000": [ciede2000]}
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(output_dir, "roberta_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    return model

# ------------------------- #
#         Main Script      #
# ------------------------- #
if __name__ == "__main__":
    # Replace with the path to your global training CSV file (with columns "Color Name", "R", "G", "B").
    csv_path = "global_training_set.csv"
    trained_model = train_final_roberta_model(csv_path,
                                                num_epochs=50,
                                                batch_size=32,
                                                lr=1e-5,
                                                val_split=0.2,
                                                patience=3,
                                                output_dir="final_model_roberta")

