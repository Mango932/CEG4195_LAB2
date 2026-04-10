from flask import Flask, request, jsonify
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image
import base64
import io
import os

# Load secrets from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-key")
port           = int(os.getenv("PORT", "5000"))
checkpoint     = os.getenv("MODEL_CHECKPOINT_PATH", "best_model.pth")
image_size     = int(os.getenv("IMAGE_SIZE", "512"))

# ── Load model once at startup ────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

# Initialize the model structure
model = deeplabv3_resnet101(weights=None)

# Update the main classifier for 2 classes (Background & House)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

# Fix: Only attempt to modify aux_classifier if it exists
if model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

if os.path.exists(checkpoint):
    # FIX: Added strict=False to ignore the 'Unexpected keys' (aux_classifier weights)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    print(f"Loaded checkpoint: {checkpoint}")
else:
    print(f"WARNING: No checkpoint found at {checkpoint} — using random weights")

model.to(device)
model.eval()
print("Model ready.")

# ── Image transform ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Aerial House Segmentation API is running!"

@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": device, "checkpoint": checkpoint})

@app.route("/segment", methods=["POST"])
def segment():
    """
    Send a POST request with a base64-encoded image:
    {
        "image": "<base64 string>"
    }
    Returns the predicted mask as a base64-encoded PNG.
    """
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided. Send a base64-encoded image in the 'image' field."}), 400

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data["image"])
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size   # (W, H) — to resize mask back

        # Run inference
        tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
        with torch.no_grad():
            output = model(tensor)["out"]                # (1, 2, H, W)
            mask   = output.argmax(dim=1).squeeze()      # (H, W)  0 or 1

        # Convert mask to image and encode as base64
        mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype("uint8"))
        mask_img = mask_img.resize(original_size, Image.NEAREST)

        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "mask":    mask_b64,
            "message": "Building pixels = white (255), background = black (0)"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=(os.getenv("FLASK_ENV") == "development"))