import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel

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
        
        # Get RGB values and normalize them to [0,1]
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
        # Remove the batch dimension (since return_tensors="pt" returns shape [1, max_length])
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
        # Map from hidden size (usually 768 for roberta-base) to 3 outputs (RGB)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask=None):
        # Get token embeddings from RoBERTa.
        # We use the first token's embedding (<s>) as a summary of the input.
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # Shape: (batch, hidden_size)
        pixel = self.fc(pooled)  # Shape: (batch, 3)
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
#         Main Script      #
# ------------------------- #
def main(csv_path, num_epochs=10, batch_size=32, lr=1e-5, val_split=0.2, output_dir="finetuned_roberta_model"):
    # Set up device and model name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "roberta-base"
    
    # Load tokenizer and text encoder from Hugging Face
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text_encoder = RobertaModel.from_pretrained(model_name)
    
    # Instantiate our RoBERTa-based model
    model = RobertaTextToPixel(text_encoder).to(device)
    
    # Ensure all layers of RoBERTa are trainable (unfrozen)
    for param in model.text_encoder.parameters():
        param.requires_grad = True
    
    # Load the dataset
    dataset = ColorDataset(csv_path, tokenizer, max_length=50)
    
    # Split dataset into training and validation sets
    total_samples = len(dataset)
    val_size = int(val_split * total_samples)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up the optimizer (often a smaller learning rate is preferred when fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    best_val_loss = float("inf")
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(output_dir, "model.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1}")
            
if __name__ == "__main__":
    # Replace with your CSV file path containing your training data
    csv_path = "sampled_training_set.csv"
    main(csv_path, num_epochs=10, batch_size=32, lr=1e-5)

