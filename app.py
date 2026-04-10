from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import torch
import os

# Load secrets from .env file — must happen before anything else
load_dotenv()

app = Flask(__name__)

# Load secrets from environment — never hard-coded
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-key")
model_name     = os.getenv("MODEL_NAME", "gpt2")
port           = int(os.getenv("PORT", "5000"))

print(f"Loading model: {model_name}")

# Load GPT-2 model & tokenizer once at startup
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model     = GPT2LMHeadModel.from_pretrained(model_name)

@app.route("/")
def home():
    return f"GPT-2 model is running! (model: {model_name})"

@app.route("/generate", methods=["POST"])
def generate():
    data   = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    input_ids     = tokenizer.encode(prompt, return_tensors="pt")
    max_new_tokens = 200
    max_length    = input_ids.shape[1] + max_new_tokens

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"generated_text": generated_text})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=(os.getenv("FLASK_ENV") == "development"))