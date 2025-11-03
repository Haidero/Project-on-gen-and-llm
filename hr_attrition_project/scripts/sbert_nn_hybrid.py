# scripts/sbert_nn_hybrid.py - Trains the Hybrid NN with Early Stopping
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../models/saved_models/best_model.pth" 
METRICS_PATH = "../results/metrics.txt" # Define the metrics file path

# 0️⃣ Create necessary directories if they don't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) 
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True) 

# 1️⃣ Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2️⃣ Define Feature Columns
# Numeric/Categorical features to be scaled and concatenated
NUM_COLS = [
    "Age", "DistanceFromHome", "MonthlyIncome",
    "YearsAtCompany", "YearsInCurrentRole",
    "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "PerformanceRating"
]

# 3️⃣ Turn structured info into text sentences for SBERT
def row_to_text(r):
    """Converts key structured features into a descriptive sentence."""
    return (f"{r['JobRole']} aged {r['Age']} works in {r['Department']} with "
            f"{r['YearsAtCompany']} years experience and job satisfaction {r['JobSatisfaction']}.")
    
df["text"] = df.apply(row_to_text, axis=1)
df["label"] = df["Attrition"].map({"Yes": 1, "No": 0})

# 4️⃣ Train/test split (stratified)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df[["text"] + NUM_COLS], df["label"], stratify=df["label"], random_state=RANDOM_STATE
)

# Separate numeric data for scaling
X_train_num = X_train_text[NUM_COLS]
X_test_num = X_test_text[NUM_COLS]
X_train_text_only = X_train_text["text"]
X_test_text_only = X_test_text["text"]

# 5️⃣ Scale Numeric Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

# 6️⃣ Generate SBERT embeddings
print("🔄 Generating SBERT embeddings. This may take a minute...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
emb_train = sbert.encode(X_train_text_only.tolist(), convert_to_numpy=True)
emb_test = sbert.encode(X_test_text_only.tolist(), convert_to_numpy=True)

# 7️⃣ Concatenate Hybrid Features
# Combine SBERT embeddings with scaled numeric features
X_train_hybrid = np.hstack([emb_train, X_train_scaled])
X_test_all = np.hstack([emb_test, X_test_scaled]) # Renamed X_test_all for clarity

# 8️⃣ Address Class Imbalance with Oversampling
# This ensures the training set is balanced before calculating class weights
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = ros.fit_resample(X_train_hybrid, y_train)

# 9️⃣ Calculate Class Weights for Loss Function (Optional but helpful)
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train_bal), y=y_train_bal
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 🔟 Define the Hybrid Neural Network (HybridNN)
class HybridNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# 1️⃣1️⃣ Initialize Model, Optimizer, and Loss
net = HybridNN(X_train_bal.shape[1]).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=5e-4) # Lower LR for smoother convergence
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

# 1️⃣2️⃣ Create DataLoader
train_ds = TensorDataset(
    torch.tensor(X_train_bal, dtype=torch.float32),
    torch.tensor(y_train_bal.values, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 1️⃣3️⃣ Training loop with Early Stopping and Best Model Saving
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # Moved to section 0

best_loss = float('inf')
patience = 5  # Stop if loss doesn't improve for 5 epochs
trigger_times = 0

print(f"🚀 Training improved hybrid model with Early Stopping on {DEVICE}...")
for epoch in range(EPOCHS):
    net.train()
    total_loss = 0
    # --- Training Loop ---
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # --- Early Stopping Check & Model Saving ---
    # Using training loss for simplicity in this script; validation loss is better practice.
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
        # 💾 SAVE: Save the model only if it's the best one so far
        torch.save(net.state_dict(), MODEL_PATH)
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
            break
            
# 1️⃣4️⃣ Load the Best Model State for Final Evaluation
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print("✅ Best model state loaded for final evaluation.")

# 1️⃣5️⃣ Final Evaluation on Unseen Test Set
print("\n📊 Evaluating model on Test Set ...")
net.eval()
with torch.no_grad():
    input_tensor = torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE)
    # Get predictions, find argmax (class), and convert to numpy
    preds = net(input_tensor).argmax(1).cpu().numpy()

report = classification_report(y_test, preds, digits=4)
print("\n--- Final Classification Report (Hybrid NN) ---")
print(report)

# 1️⃣6️⃣ Save metrics
with open(METRICS_PATH, "a") as f:
    f.write("\n\n--- Start of Hybrid NN Results ---\n")
    f.write("\nHybrid NN (SBERT + Numeric) with Early Stopping\n")
    f.write(report + "\n")
    f.write("--- End of Hybrid NN Results ---\n\n")

# 📢 Added this line to show the exact location where results were saved.
print(f"✅ Final test results appended to: {os.path.abspath(METRICS_PATH)}")

# Optionally, you can now also generate the confusion matrix here if you like
# ConfusionMatrixDisplay.from_predictions(y_test, preds)
# plt.title("Hybrid Model Test Set")
# plt.show()
