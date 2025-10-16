# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import KeypointSequenceDataset
from model import SimpleTransformerClassifier
from tqdm import tqdm
import os

SEQ_DIR = "data/sequences"
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
MODEL_OUT = "models/model.pth"

def main():
    ds = KeypointSequenceDataset(SEQ_DIR)
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    sample_x, _ = ds[0]
    T, D = sample_x.shape
    model = SimpleTransformerClassifier(input_dim=D, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=LR)

    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}"):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
        acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print("Saved best model to", MODEL_OUT)

if __name__ == "__main__":
    main()
