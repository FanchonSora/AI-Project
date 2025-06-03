import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Use a relative path based on the current file location
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
H5_PATH = os.path.join(PROJECT_ROOT, "Galaxy10_DECals.h5")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "galaxy10_b0(1).pth")  # Checkpoint stays in backend folder

if not os.path.exists(H5_PATH):
    sys.exit(f"❌ HDF5 file not found: {H5_PATH}")
else:
    print("✅ Found:", H5_PATH)

IMG_SIZE = 128
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
AUGMENT = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Dataset definition
class Galaxy10H5(Dataset):
    def __init__(self, h5_path, indices=None, transform=None):
        self.h5 = h5py.File(h5_path, "r")
        self.imgs = self.h5["images"]  # Shape: (N, H, W, C)
        self.labs = self.h5["ans"]     # Shape: (N,)
        self.idxs = np.arange(len(self.imgs)) if indices is None else indices
        self.tfm = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        img = self.imgs[self.idxs[i]]
        lab = int(self.labs[self.idxs[i]])
        if self.tfm:
            img = self.tfm(img)
        return img, lab

def main():
    # Prepare dataset and dataloaders
    full_ds = Galaxy10H5(H5_PATH, transform=AUGMENT)
    N = len(full_ds)
    n_val = int(0.10 * N)
    n_test = int(0.10 * N)
    n_train = N - n_val - n_test
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)
    
    BATCH = 64
    # Set NUM_WORKERS=0 for Windows/h5py compatibility
    NUM_WORKERS = 0  
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    NUM_CLASSES = 10
    # Use EfficientNet-B0 from torchvision with modified classifier to match 10 classes
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler()
    
    # Training loop
    EPOCHS = 7
    PATIENCE = 5
    best_val_acc = 0
    patience_cnt = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, correct, total = 0, 0, 0
        torch.set_grad_enabled(train)
        loop = tqdm(loader, leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad(), autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            loop.set_postfix(loss=loss.item())
        return total_loss / total, correct / total
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.3f} val_loss={val_loss:.3f} "
              f"train_acc={tr_acc:.3%} val_acc={val_acc:.3%}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({"model": model.state_dict(),
                        "acc": best_val_acc,
                        "epoch": epoch,
                        "img_size": IMG_SIZE}, SAVE_PATH)
            print("✅ Saved new best model")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("⏹ Early-stopping triggered")
                break

    # Test evaluation
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            logits = model(x.to(device))
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = (all_preds == all_labels).mean() * 100
    print("Test accuracy:", accuracy, "%")
    print(classification_report(all_labels, all_preds,
          target_names=[f"class_{i}" for i in range(NUM_CLASSES)]))
    
    # (Optional) Save learning curve plots
    plt.figure(figsize=(5,4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.figure(figsize=(5,4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()

app = Flask(__name__)
CORS(app)  # enable CORS if your frontend is on a different origin

# Reinitialize the model for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
inference_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
inference_model.classifier[1] = torch.nn.Linear(inference_model.classifier[1].in_features, NUM_CLASSES)
inference_model = inference_model.to(device)
# Load saved checkpoint (assume training has been run prior)
if os.path.exists(SAVE_PATH):
    ckpt = torch.load(SAVE_PATH, map_location=device)
    inference_model.load_state_dict(ckpt["model"])
    print("✅ Inference model loaded from checkpoint")
else:
    print("⚠️ Checkpoint not found. Please run training first.")
inference_model.eval()

# Define an inference transform (use a simple deterministic transform)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

@app.route("/predict1", methods=["POST"])
def predict1():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Apply the inference transform 
    img_tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = inference_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return jsonify({
        "label": pred,
        "confidence": confidence
    })

if __name__ == "__main__":
    # Run Flask server if "predict" argument is provided; otherwise, run training.
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        main()