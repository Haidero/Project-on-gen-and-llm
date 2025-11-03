# scripts/sbert_nn.py  (Improved, balanced)
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Turn structured data into short text
def row_to_text(row):
    return f"{row['JobRole']} in {row['Department']} with {row['YearsAtCompany']} years and performance rating {row['PerformanceRating']}."
df['text'] = df.apply(row_to_text, axis=1)
df['label'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], stratify=df['label'], random_state=42
)

# 4. Encode with Sentence-BERT
print("🔄 Generating SBERT embeddings ...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
emb_train = sbert.encode(X_train.tolist(), show_progress_bar=True)
emb_test  = sbert.encode(X_test.tolist(),  show_progress_bar=True)

# 5. Oversample minority class
ros = RandomOverSampler(random_state=42)
emb_train_bal, y_train_bal = ros.fit_resample(emb_train, y_train)
print(f"✅ After oversampling: {np.bincount(y_train_bal)}")

# 6. Compute class weights for loss function
classes = np.unique(y_train_bal)
weights = compute_class_weight("balanced", classes=classes, y=y_train_bal)
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Class weights:", class_weights)

# 7. Build simple feed-forward NN
class FFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = FFNN(emb_train.shape[1]).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 8. Prepare DataLoader
train_ds = TensorDataset(
    torch.tensor(emb_train_bal, dtype=torch.float32),
    torch.tensor(y_train_bal.values, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 9. Train network
print("🚀 Training balanced network ...")
for epoch in range(20):  # more epochs
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}/20 - Loss: {total_loss/len(train_loader):.4f}")

# 10. Evaluate
print("\n📊 Evaluating model ...")
net.eval()
with torch.no_grad():
    preds = net(torch.tensor(emb_test, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()

report = classification_report(y_test, preds, digits=4)
print(report)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Balanced SBERT + Neural Network")
plt.show()

# 11. Save metrics
with open("results/metrics.txt", "a") as f:
    f.write("\nBalanced SBERT + Neural Network\n")
    f.write(report + "\n")

print("✅ Results appended to results/metrics.txt")
