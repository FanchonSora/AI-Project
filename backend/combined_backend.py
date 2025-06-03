# combined_backend.py
import io, torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
import timm                         # for EfficientNet-B2 (new model)

DEVICE, NUM_CLASSES = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    10,
)

# ─── Model wrappers ────────────────────────────────────────────────────────────
class NewGalaxyClassifier:
    """EfficientNet-B2, weights = galaxy10_ema.pth"""
    def __init__(self, ckpt="galaxy10_ema.pth"):
        self.model = timm.create_model(
            "efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        self.model.eval()
        self.tfm = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, img: Image.Image):
        x = self.tfm(img).unsqueeze(0).to(DEVICE)
        p = torch.softmax(self.model(x), 1)[0]
        idx = int(torch.argmax(p))
        return idx, float(p[idx])

class OldGalaxyClassifier:
    """EfficientNet-B0, weights = galaxy10_b0.pth (trained locally)"""
    def __init__(self, ckpt="galaxy10_b0.pth", img_size=128):
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        self.model = m.to(DEVICE)
        self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        self.model.eval()
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def __call__(self, img: Image.Image):
        x = self.tfm(img).unsqueeze(0).to(DEVICE)
        p = torch.softmax(self.model(x), 1)[0]
        idx = int(torch.argmax(p))
        return idx, float(p[idx])

# ─── Ensemble logic ────────────────────────────────────────────────────────────
LABELS = {
    0: "Disturbed Galaxy", 1: "Merging Galaxy", 2: "Round Smooth Galaxy",
    3: "Intermediate Smooth Galaxy", 4: "Cigar-shaped Smooth Galaxy",
    5: "Barred Spiral Galaxy", 6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral", 8: "Edge-on Galaxy (no bulge)",
    9: "Edge-on Galaxy (with bulge)",
}

class Ensemble:
    def __init__(self):
        self.new, self.old = NewGalaxyClassifier(), OldGalaxyClassifier()

    def predict(self, img: Image.Image):
        idx_new, conf_new = self.new(img)
        idx_old, conf_old = self.old(img)

        # Agreement → average confidence; Disagreement → trust old model
        if idx_new == idx_old:
            idx, conf, src = idx_new, (conf_new + conf_old) / 2, "both"
        else:
            idx, conf, src = idx_old, conf_old, "old_model"

        return {
            "label_id": idx,
            "label": LABELS[idx],
            "confidence": conf,
            "source": src,
        }

app, ensemble = Flask(__name__), Ensemble()
CORS(app, resources={r"/predict2": {"origins": "*"}})

@app.route("/predict2", methods=["POST"])
def predict2():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    return jsonify(ensemble.predict(img))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
