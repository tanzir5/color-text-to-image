import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from skimage import color  # For CIEDE2000 computation
from scipy.stats import pearsonr

# ------------------------- #
#    SinglePixelDataset     #
# ------------------------- #
class SinglePixelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_length=50):
        # Assumes CSV has columns: "color_name", "r", "g", "b", "abstraction", "ambiguity"
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # We ignore the extra columns for inference, but they remain in self.df.
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Use the "color_name" column for the text prompt.
        prompt = self.df.iloc[idx]['color_name']
        # Get gold RGB values (assumed to be in 0-255) and normalize them.
        r = self.df.iloc[idx]['r']
        g = self.df.iloc[idx]['g']
        b = self.df.iloc[idx]['b']
        target_pixel = torch.tensor([r, g, b], dtype=torch.float32) / 255.0
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids.squeeze(0)
        return input_ids, target_pixel

# ------------------------- #
#   CIEDE2000 Computation   #
# ------------------------- #
def compute_ciede2000(y_true, y_pred):
    """
    Compute CIEDE2000 color difference per color.
    Inputs (y_true and y_pred) are assumed to be in [0,255].
    """
    y_true = np.array(y_true) / 255.0
    y_pred = np.array(y_pred) / 255.0
    y_true_lab = color.rgb2lab(y_true.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred.reshape(-1, 1, 3)).reshape(-1, 3)
    return color.deltaE_ciede2000(y_true_lab, y_pred_lab)

# ------------------------- #
# Evaluation Function       #
# ------------------------- #
def evaluate_final_model_on_test(
    model,
    test_csv_path,
    model_name,
    tokenizer,
    device,
    batch_size=32
):
    """
    Evaluate a trained model (RoBERTa or CLIP based) on a global test set.
    
    The test CSV is assumed to have the following columns (all lower-case):
        - color_name : text prompt.
        - r, g, b   : gold RGB values (0-255).
        - abstraction : abstraction score (0-1).
        - ambiguity   : ambiguity score (0-1).
    
    The function performs:
      1. Inference to obtain predictions (scaled to [0,255]).
      2. Per-color error metrics: RMSE, MAE, and CIEDE2000.
      3. Global metrics (average errors).
      4. Pearson correlation of each error metric with abstraction and ambiguity.
      5. Grouped (binned) analysis by ambiguity and abstraction levels.
      6. Saves per-color errors and grouped analysis to CSV files.
    
    Returns a dictionary with overall metrics and DataFrames of the analyses.
    """
    # ---- Step 1: Load test data & create DataLoader ---- #
    dataset = SinglePixelDataset(test_csv_path, tokenizer, max_length=50)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for input_ids, target in loader:
            input_ids = input_ids.to(device)
            preds = model(input_ids)  # Model outputs in [0,1]
            all_preds.append(preds.cpu())
            all_targets.append(target)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()   # shape: (N, 3)
    all_targets = torch.cat(all_targets, dim=0).numpy() # shape: (N, 3)
    min_values = np.min(all_preds, axis=0)
    max_values = np.max(all_preds, axis=0)

    print("Minimum values for (R, G, B):", min_values)
    print("Maximum values for (R, G, B):", max_values)
    
    # ---- Step 2: Scale predictions/targets to [0,255] ---- #
    preds_scaled = all_preds.clip(0, 1) * 255.0
    targets_scaled = all_targets.clip(0, 1) * 255.0
    
    # ---- Step 3: Compute per-color error metrics ---- #
    # RMSE and MAE per sample (computed over RGB channels)
    rmse_per_color = np.sqrt(np.mean((targets_scaled - preds_scaled) ** 2, axis=1))
    mae_per_color = np.mean(np.abs(targets_scaled - preds_scaled), axis=1)
    ciede2000_per_color = compute_ciede2000(targets_scaled, preds_scaled)
    
    # Global metrics (average over all samples)
    rmse_global = np.mean(rmse_per_color)
    mae_global = np.mean(mae_per_color)
    ciede2000_global = np.mean(ciede2000_per_color)
    
    print(f"\n✅ Overall Test Set Evaluation for {model_name}:")
    print(f"RMSE: {rmse_global:.2f}, MAE: {mae_global:.2f}, Mean CIEDE2000: {ciede2000_global:.2f}")
    
    # ---- Step 4: Save per-color errors to CSV ---- #
    test_df = pd.read_csv(test_csv_path)
    # Expecting test_df to have 'color_name'
    results_df = pd.DataFrame({
        'color_name': test_df['color_name'],
        'gold_r': targets_scaled[:, 0],
        'gold_g': targets_scaled[:, 1],
        'gold_b': targets_scaled[:, 2],
        'pred_r': preds_scaled[:, 0],
        'pred_g': preds_scaled[:, 1],
        'pred_b': preds_scaled[:, 2],
        'rmse': rmse_per_color,
        'mae': mae_per_color,
        'ciede2000': ciede2000_per_color
    })
    results_csv = f"results/{model_name}_test_results.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"Per-color results saved to {results_csv}")
    
    # ---- Step 5: Correlation Analysis ---- #
    # Load abstraction and ambiguity scores from test CSV.
    # (Assuming these columns are named 'abstraction' and 'ambiguity'.)
    scores_df = test_df[['color_name', 'abstraction', 'ambiguity']].dropna()
    merged_df = results_df.merge(scores_df, on='color_name')
    
    # Compute Pearson correlations for each metric.
    for metric in ['rmse', 'mae', 'ciede2000']:
        ambiguity_corr, _ = pearsonr(merged_df['ambiguity'], merged_df[metric])
        abstraction_corr, _ = pearsonr(merged_df['abstraction'], merged_df[metric])
        print(f"{metric.upper()}: Ambiguity Corr={ambiguity_corr:.3f}, Abstraction Corr={abstraction_corr:.3f}")
    
    # ---- Step 6: Grouped Analysis by Ambiguity ---- #
    bins = [0, 0.33, 0.66, 1]
    labels = ["Low", "Medium", "High"]
    merged_df["ambiguity_level"] = pd.cut(merged_df["ambiguity"], bins=bins, labels=labels)
    grouped_errors_amb = merged_df.groupby("ambiguity_level").agg(
        {"rmse": ["mean", "std"], "mae": ["mean", "std"], "ciede2000": ["mean", "std"]}
    ).reset_index()
    grouped_errors_amb_csv = f"results/{model_name}_grouped_error_analysis_ambiguity.csv"
    grouped_errors_amb.to_csv(grouped_errors_amb_csv, index=False)
    
    print("\n📊 Grouped Error Analysis by Ambiguity Level:")
    print(grouped_errors_amb)
    
    # ---- Step 7: Grouped Analysis by Abstraction ---- #
    merged_df["abstraction_level"] = pd.cut(merged_df["abstraction"], bins=bins, labels=labels)
    grouped_errors_abs = merged_df.groupby("abstraction_level").agg(
        {"rmse": ["mean", "std"], "mae": ["mean", "std"], "ciede2000": ["mean", "std"]}
    ).reset_index()
    grouped_errors_abs_csv = f"results/{model_name}_grouped_error_analysis_abstraction.csv"
    grouped_errors_abs.to_csv(grouped_errors_abs_csv, index=False)
    
    print("\n📊 Grouped Error Analysis by Abstraction Level:")
    print(grouped_errors_abs)
    
    # ---- Step 8: Return all metrics ---- #
    return {
        'overall_metrics': {'rmse': rmse_global, 'mae': mae_global, 'ciede2000': ciede2000_global},
        'per_color_results': results_df,
        'merged_results': merged_df,
        'grouped_errors_ambiguity': grouped_errors_amb,
        'grouped_errors_abstraction': grouped_errors_abs
    }

# ------------------------- #
#       Example Usage       #
# ------------------------- #
if __name__ == "__main__":
    # For demonstration, choose model_type: "roberta" or "clip"
    model_type = "clip"  # Change as needed.
    
    if model_type == "roberta":
        from transformers import RobertaTokenizer, RobertaModel
        model_name = "final_model_roberta"
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        text_encoder = RobertaModel.from_pretrained("roberta-base")
        # Define the model architecture used during training.
        class RobertaTextToPixel(nn.Module):
            def __init__(self, text_encoder: RobertaModel):
                super().__init__()
                self.text_encoder = text_encoder
                self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)
            def forward(self, input_ids):
                outputs = self.text_encoder(input_ids=input_ids)
                pooled = outputs.last_hidden_state[:, 0, :]
                pixel = self.fc(pooled)
                return pixel
        model = RobertaTextToPixel(text_encoder)
        from_path = "final_model_roberta/model.pt"  # adjust if needed
        model.load_state_dict(torch.load(from_path, map_location="cpu"))
    else:
        from transformers import CLIPTokenizer, CLIPTextModel
        model_name = "final_model_clip"
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        class CLIPTextToPixel(nn.Module):
            def __init__(self, text_encoder: CLIPTextModel):
                super().__init__()
                self.text_encoder = text_encoder
                self.fc = nn.Linear(self.text_encoder.config.hidden_size, 3)
            def forward(self, input_ids):
                outputs = self.text_encoder(input_ids=input_ids)
                pooled = outputs.last_hidden_state.mean(dim=1)
                pixel = self.fc(pooled)
                return pixel
        model = CLIPTextToPixel(text_encoder)
        from_path = "final_model_clip/model.pt"  # adjust if needed
        model.load_state_dict(torch.load(from_path, map_location="cpu"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Path to your global test CSV file.
    # The CSV should include columns: "color_name", "r", "g", "b", "abstraction", "ambiguity"
    test_csv_path = "global_test_set.csv"
    
    metrics = evaluate_final_model_on_test(model, test_csv_path, model_name, tokenizer, device, batch_size=32)

