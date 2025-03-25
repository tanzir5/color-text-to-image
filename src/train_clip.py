import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torchvision.transforms as transforms
from skimage import color  # For CIEDE2000 computation
import numpy as np

# ------------------------- #
#      SinglePixelDataset    #
# ------------------------- #
class SinglePixelDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=50):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Transformation to convert any image into a single pixel.
        # (This isn't used in this version because we directly read RGB values.)
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
class CLIPTextToPixel(nn.Module):
    """
    Uses CLIP's text encoder to get a text embedding and then passes that
    embedding through a linear layer to output three values (R, G, B).
    """
    def __init__(self, text_encoder: CLIPTextModel):
        super().__init__()
        self.text_encoder = text_encoder
        # Linear layer mapping from hidden_size to 3 (RGB)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)
    
    def forward(self, input_ids):
        # Get text embeddings from the CLIP text encoder.
        outputs = self.text_encoder(input_ids=input_ids)
        # Pool embeddings by taking the mean over tokens.
        pooled = outputs.last_hidden_state.mean(dim=1)  # shape: (batch, hidden_size)
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
def train_final_clip_model(csv_path, num_epochs=50, batch_size=32, lr=1e-5, val_split=0.2, patience=3, output_dir="final_model_clip"):
    """
    Train the final CLIP-based text-to-RGB model on the entire global training dataset.
    Uses a hold-out validation split and early stopping to avoid overfitting.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-large-patch14"
    
    # Load tokenizer and pre-trained CLIP text encoder.
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name)
    
    # Instantiate the model.
    model = CLIPTextToPixel(text_encoder).to(device)
    # All layers are unfrozen by default.
    
    # Load the global dataset using our provided SinglePixelDataset.
    dataset = SinglePixelDataset(csv_path, tokenizer, max_length=50)
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders.
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
        train_loss = 0.0
        for input_ids, target in train_loader:
            input_ids = input_ids.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
        train_loss /= train_size
        
        # Validation phase.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target in val_loader:
                input_ids = input_ids.to(device)
                target = target.to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, target)
                val_loss += loss.item() * input_ids.size(0)
        val_loss /= val_size
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Early stopping: save best model and monitor improvements.
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
    
    # Optionally compute final metrics on the validation set.
    rmse, mae, ciede2000 = compute_metrics(model, val_loader, device)
    print(f"Final Validation Metrics: RMSE = {rmse:.2f}, MAE = {mae:.2f}, CIEDE2000 = {ciede2000:.2f}")
    
    return model

# ------------------------- #
#         Main Script      #
# ------------------------- #
if __name__ == "__main__":
    # Replace with the path to your global training CSV file (with columns "Color Name", "R", "G", "B").
    csv_path = "global_training_set.csv"
    trained_model = train_final_clip_model(csv_path,
                                             num_epochs=50,
                                             batch_size=32,
                                             lr=1e-5,
                                             val_split=0.2,
                                             patience=3,
                                             output_dir="final_model_clip")

