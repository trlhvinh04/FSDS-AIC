import torch
import time
import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CAPTION_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ==============================================================================
# LOAD MODELS (Once, globally)
# ==============================================================================
print("📦 Loading models...")
feature_extractor = AutoFeatureExtractor.from_pretrained(CAPTION_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_NAME)
caption_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)
caption_model.eval()

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
print("✅ Models loaded.")


# ==============================================================================
# FUNCTION TO GENERATE CAPTIONS AND EMBEDDINGS
# ==============================================================================
def generate_caption_embeddings(image_folder: str, batch_size: int = BATCH_SIZE):
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"🖼️ Found {len(image_files)} images in {image_folder}")

    captions = []
    embeddings = []
    

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = [Image.open(os.path.join(image_folder, f)).convert("RGB") for f in batch_files]

        inputs = feature_extractor(images=batch_images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = caption_model.generate(**inputs, num_beams=1, max_new_tokens=30)
            batch_captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            batch_embeddings = embedder.encode(batch_captions)

        embeddings.extend(batch_embeddings)
        
    return embeddings


