import io, torch, timm
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS 
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10                 # Galaxy10 có 10 lớp

# Khởi tạo đúng kiến trúc EfficientNet-B2
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
model.to(DEVICE)
# Nạp checkpoint
ckpt = torch.load("galaxy10_ema.pth", map_location=DEVICE)
state_dict = ckpt.get("model", ckpt)         
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("✓ loaded – missing:", len(missing), "unexpected:", len(unexpected))
model.eval()

# Transform (giống khi train)
transform = transforms.Compose([
    transforms.Resize((260, 260)),          # B2 thường training 260×260; resize 224×224 vẫn OK
    transforms.CenterCrop((224, 224)),      # nếu muốn đúng hẳn 224×224
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict2": {"origins": "http://localhost:3000"}})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file in request!")
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    print("File received:", file.filename)
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        return jsonify({"error": str(e)}), 400
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        conf = float(probs[0, pred_id])
    # Mapping từ ID sang tên lớp
    LABEL_MAP = {
        0: "Disturbed Galaxy",
        1: "Merging Galaxy",
        2: "Round Smooth Galaxy",
        3: "Intermediate Smooth Galaxy",
        4: "Cigar-shaped Smooth Galaxy",
        5: "Barred Spiral Galaxy",
        6: "Unbarred Tight Spiral",
        7: "Unbarred Loose Spiral",
        8: "Edge-on Galaxy (no bulge)",
        9: "Edge-on Galaxy (with bulge)",
    }
    pred_name = LABEL_MAP.get(pred_id, f"class_{pred_id}")
    return jsonify({
        "label_id":   pred_id,
        "label":      pred_name,
        "confidence": conf
    }), 200
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)