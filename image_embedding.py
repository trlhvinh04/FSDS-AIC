import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
import clip
from PIL import Image, UnidentifiedImageError
import faiss
import kagglehub 

BATCH_SIZE = 64
USE_GPU_FAISS = True
FAISS_GPU_ID = 0

SAVE_EMB_PATH = "clip_embeds.npy"
SAVE_PATHS_PATH = "clip_paths.txt"
SAVE_FAISS_PATH = "clip_faiss.index"

def get_base_dir_from_kaggle(dataset_id: str) -> str:
    """
    Download dataset via kagglehub and try to find the keyframes root.
    """
    root = kagglehub.dataset_download(dataset_id)
    candidates = [
        os.path.join(root, "version", "1", "keyframes"),
        os.path.join(root, "keyframes"),
        root,
    ]
    for c in candidates:
        if os.path.isdir(c):
            if glob.glob(os.path.join(c, "L06_V*")):
                return c
    matches = glob.glob(os.path.join(root, "**", "L06_V*"), recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    return root

def collect_image_paths(base_dir: str,
                        patterns=("*.jpg", "*.jpeg", "*.png"),
                        recursive=True) -> List[str]:
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(base_dir, "**", pat), recursive=recursive))
    out = [p for p in out if os.path.isfile(p)]
    out.sort()
    return out

device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> Using device:", device)
model, preprocess = clip.load("ViT-L/14@336px", device=device) # Hoặc là dùng "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-L/14", "ViT-B/16", "ViT-B/32", "ViT-L/14@336px"
model.eval()

def embed_images_batch(paths: List[str],
                       batch_size: int = 64,
                       device: str = device) -> np.ndarray:
    feats = []
    valid_paths = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        keep_idx = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Lỗi đọc ảnh {p}: {e}. Bỏ qua.")
                continue
            imgs.append(preprocess(img))
            keep_idx.append(j)
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = model.encode_image(batch)            # (B,512)
            f = f / f.norm(dim=-1, keepdim=True)     # cosine
        feats.append(f.cpu().numpy().astype(np.float32))
        # lưu các path hợp lệ
        valid_paths.extend([batch_paths[j] for j in keep_idx])
    if not feats:
        raise RuntimeError("Không embed được ảnh nào!")
    embs = np.concatenate(feats, axis=0)
    return embs, valid_paths


def embed_text(text: str, device: str = device) -> np.ndarray:
    toks = clip.tokenize([text]).to(device)
    with torch.no_grad():
        f = model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype(np.float32).squeeze(0)  # (512,)

def build_faiss_cpu(embs: np.ndarray) -> faiss.Index:
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)  # cosine (vectors normalized)
    idx.add(embs)
    return idx

def maybe_to_gpu(index_cpu: faiss.Index,
                 use_gpu: bool = USE_GPU_FAISS,
                 gpu_id: int = FAISS_GPU_ID):
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        idx_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
        return idx_gpu
    return index_cpu


def save_embeddings(embs: np.ndarray,
                    paths: List[str],
                    emb_path: str = SAVE_EMB_PATH,
                    paths_path: str = SAVE_PATHS_PATH,
                    faiss_path: str = SAVE_FAISS_PATH,
                    index_cpu_obj: Optional[faiss.Index] = None):
    np.save(emb_path, embs)
    with open(paths_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")
    if index_cpu_obj is not None:
        faiss.write_index(index_cpu_obj, faiss_path)
    print(f"[INFO] Saved embeddings -> {emb_path}")
    print(f"[INFO] Saved paths      -> {paths_path}")
    if index_cpu_obj is not None:
        print(f"[INFO] Saved FAISS CPU index -> {faiss_path}")

def load_embeddings(emb_path: str = SAVE_EMB_PATH,
                    paths_path: str = SAVE_PATHS_PATH,
                    faiss_path: str = SAVE_FAISS_PATH,
                    use_gpu: bool = USE_GPU_FAISS,
                    gpu_id: int = FAISS_GPU_ID):
    embs = np.load(emb_path)
    with open(paths_path, "r", encoding="utf-8") as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    idx_cpu = faiss.read_index(faiss_path)
    idx = maybe_to_gpu(idx_cpu, use_gpu, gpu_id)
    return embs, paths, idx

def make_search_fn(index, image_paths: List[str]):
    """
    Trả về hàm search_text() đóng gói index + paths.
    """
    def search_text(query: str, topk: int = 5) -> List[Tuple[str, float]]:
        qv = embed_text(query)[None, :]          # (1,512)
        D, I = index.search(qv, topk)
        results = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(image_paths):
                results.append((image_paths[idx], float(score)))
        return results
    return search_text

if __name__ == "__main__":
    # ---- tải dataset Kaggle ----
    DATASET_ID = "phucnguyenchau/keyframes-l06"
    print(f"[INFO] Loading dataset: {DATASET_ID}")
    path = kagglehub.dataset_download(DATASET_ID)
    # print(f"[INFO] Kaggle local path: {path}")

    # ---- xác định BASE_DIR chứa L06_Vxxx ----
    BASE_DIR = get_base_dir_from_kaggle(DATASET_ID)
    print(f"[INFO] Found dataset: {BASE_DIR}")

    # ---- thu thập ảnh ----
    image_paths = collect_image_paths(BASE_DIR)
    print(f"[INFO] Total image found: {len(image_paths)}")
    if not image_paths:
        raise FileNotFoundError("No image found")

    # ---- embed ảnh ----
    print("[INFO] Embedding images with CLIP...")
    embs, image_paths = embed_images_batch(image_paths, BATCH_SIZE, device)
    print(f"[INFO] Embedding shape: {embs.shape}")  # (N,512)

    # ---- build FAISS ----
    index_cpu = build_faiss_cpu(embs)
    print(f"[INFO] FAISS CPU index contains {index_cpu.ntotal} vectors.")

    index = maybe_to_gpu(index_cpu, USE_GPU_FAISS, FAISS_GPU_ID)
    if index is not index_cpu:
        print("[INFO] FAISS index has been moved to GPU.")
    else:
        print("[INFO] Using FAISS CPU index.")

    # ---- lưu (optional) ----
    save_embeddings(embs, image_paths, index_cpu_obj=index_cpu)

    # ---- search demo ----
    search_text = make_search_fn(index, image_paths)
    query = "một chiếc xe ô tô màu đỏ đang dừng bên lề đường"
    results = search_text(query, topk=5)
    print(f"\n[RESULT] Query: {query!r}")
    for path_i, score in results:
        print(f"{score: .4f}  {path_i}")
