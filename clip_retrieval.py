"""Shared data, CLIP inference, and retrieval evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "imagenette2-160"
CACHE_DIR = ROOT / "cache"
ARCHIVE_PATH = DATA_DIR / "imagenette2-160.tgz"
IMAGE_LIST_PATH = CACHE_DIR / "image_manifest.json"
FEATURE_CACHE_PATH = CACHE_DIR / "image_features.npz"
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
MODEL_ID = "openai/clip-vit-base-patch32"
LOCAL_MODEL_DIR = DATA_DIR / "clip-vit-base-patch32"
MODEL_PATH = os.environ.get(
    "CLIP_MODEL_PATH",
    str(LOCAL_MODEL_DIR if (LOCAL_MODEL_DIR / "config.json").exists() else MODEL_ID),
)
SEED = 42
IMAGES_PER_CLASS = 100
BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32


def prepare_dataset() -> tuple[list[Path], list[str]]:
    """Download if needed and reproduce the fixed 1,000-image gallery."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not IMAGE_DIR.exists():
        if not ARCHIVE_PATH.exists():
            print(f"下载 Imagenette：{DATA_URL}")
            with urllib.request.urlopen(DATA_URL, timeout=60) as response, ARCHIVE_PATH.open("wb") as target:
                total = int(response.headers.get("Content-Length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc="下载图片集") as progress:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        progress.update(len(chunk))
        print("解压图片集…")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            archive.extractall(DATA_DIR, filter="data")

    train_dir = IMAGE_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"没有找到 Imagenette 训练目录：{train_dir}")

    rng = random.Random(SEED)
    selected: list[Path] = []
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        candidates = sorted(
            path for path in class_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(candidates) < IMAGES_PER_CLASS:
            raise ValueError(f"类别 {class_dir.name} 只有 {len(candidates)} 张图片")
        selected.extend(rng.sample(candidates, IMAGES_PER_CLASS))

    image_paths = sorted(selected)
    if len(image_paths) != 1000:
        raise ValueError(f"预期 1000 张图片，实际找到 {len(image_paths)} 张")
    image_manifest = [path.relative_to(ROOT).as_posix() for path in image_paths]
    IMAGE_LIST_PATH.write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已准备 {len(image_paths)} 张图片，来自 {len({p.parent.name for p in image_paths})} 个类别。")
    print(f"运行设备：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    return image_paths, image_manifest


def load_model() -> tuple[torch.device, Any, CLIPModel]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型 {MODEL_PATH} 到 {device}…")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = CLIPModel.from_pretrained(MODEL_PATH).to(device).eval()
    except OSError as error:
        raise RuntimeError(
            "无法加载 CLIP 模型。请检查 Hugging Face 网络，或设置 CLIP_MODEL_PATH 指向已下载的模型目录。"
        ) from error
    print(f"模型已就绪；特征维度：{model.config.projection_dim}")
    return device, processor, model


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _features(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    elif isinstance(output, tuple):
        output = output[0]
    return output


def encode_image_gallery(
    image_paths: list[Path], image_manifest: list[str], device: torch.device,
    processor: Any, model: CLIPModel,
) -> torch.Tensor:
    file_facts = []
    for path in image_paths:
        stat = path.stat()
        file_facts.append((path.relative_to(ROOT).as_posix(), stat.st_size, stat.st_mtime_ns))
    signature = hashlib.sha256(
        json.dumps({"model": str(MODEL_PATH), "images": file_facts}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if FEATURE_CACHE_PATH.exists():
        with np.load(FEATURE_CACHE_PATH, allow_pickle=False) as saved:
            valid = str(saved["signature"].item()) == signature
            if valid:
                cached_paths = saved["paths"].tolist()
                valid = cached_paths == image_manifest
                if valid:
                    image_features = torch.from_numpy(saved["features"].copy()).to(device)
    else:
        valid = False

    if valid:
        print(f"已加载缓存：{FEATURE_CACHE_PATH}")
    else:
        batches = []
        for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="编码图片"):
            images = []
            for path in image_paths[start:start + BATCH_SIZE]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                features = _features(model.get_image_features(**inputs))
            batches.append(normalize(features).float().cpu())
        image_features = torch.cat(batches, dim=0).to(device)
        np.savez_compressed(
            FEATURE_CACHE_PATH, features=image_features.cpu().numpy(),
            paths=np.asarray(image_manifest), signature=np.asarray(signature),
        )
        print(f"图片向量已缓存到 {FEATURE_CACHE_PATH}")

    expected_shape = (len(image_paths), model.config.projection_dim)
    if image_features.shape != expected_shape:
        raise ValueError(f"图片特征维度错误：预期 {expected_shape}，实际 {tuple(image_features.shape)}")
    if not torch.isfinite(image_features).all():
        raise ValueError("图片特征包含非有限值")
    if not torch.allclose(image_features.norm(dim=1), torch.ones(len(image_paths), device=device), atol=1e-4):
        raise ValueError("图片特征尚未归一化")
    print(f"图片特征矩阵：{tuple(image_features.shape)}；每行都是单位向量。")
    return image_features


def encode_texts(queries: list[str], device: torch.device, processor: Any, model: CLIPModel) -> torch.Tensor:
    chunks = []
    for start in range(0, len(queries), TEXT_BATCH_SIZE):
        inputs = processor(
            text=queries[start:start + TEXT_BATCH_SIZE],
            return_tensors="pt", padding=True, truncation=True,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            chunks.append(normalize(_features(model.get_text_features(**inputs))).float())
    return torch.cat(chunks, dim=0)


def rank_queries(
    queries: list[str], image_features: torch.Tensor, image_paths: list[Path],
    device: torch.device, processor: Any, model: CLIPModel, top_k: int = 10,
) -> list[list[dict[str, Any]]]:
    if not 1 <= top_k <= len(image_paths):
        raise ValueError(f"top_k 必须在 1 到 {len(image_paths)} 之间。")
    text_features = encode_texts(queries, device, processor, model)
    similarities = text_features @ image_features.T
    results = []
    # Python's stable sort with an explicit index key gives deterministic gallery-order ties.
    for scores in similarities.cpu().tolist():
        order = sorted(range(len(image_paths)), key=lambda index: (-scores[index], index))[:top_k]
        results.append([
            {"path": image_paths[index], "score": float(scores[index])}
            for index in order
        ])
    return results


def validate_ground_truth(records: Any, image_manifest: list[str]) -> list[dict[str, str]]:
    if not isinstance(records, list) or not records:
        raise ValueError("Ground truth 必须是非空 JSON 数组。")
    ids: set[str] = set()
    queries: set[str] = set()
    gallery = set(image_manifest)
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index + 1} 条标注必须是 JSON 对象。")
        missing = {"query_id", "query", "image_path"} - record.keys()
        if missing:
            raise ValueError(f"第 {index + 1} 条标注缺少字段：{', '.join(sorted(missing))}")
        query_id, query, image_path = (record[key] for key in ("query_id", "query", "image_path"))
        if not all(isinstance(value, str) for value in (query_id, query, image_path)):
            raise ValueError(f"第 {index + 1} 条标注的 query_id、query、image_path 必须是字符串。")
        query_id, query, image_path = query_id.strip(), query.strip(), image_path.strip()
        if not query_id:
            raise ValueError(f"第 {index + 1} 条标注的 query_id 不能为空。")
        if not query:
            raise ValueError(f"Query {query_id} 不能为空。")
        if query_id in ids:
            raise ValueError(f"重复的 query_id：{query_id}")
        if query.casefold() in queries:
            raise ValueError(f"重复的 query：{query}")
        if image_path not in gallery:
            raise ValueError(f"Query {query_id} 的目标图片不在当前图库中：{image_path}")
        ids.add(query_id)
        queries.add(query.casefold())
        normalized.append({"query_id": query_id, "query": query, "image_path": image_path})
    return normalized


def calculate_metrics(rank: int, gallery_size: int) -> dict[str, float | int]:
    if not 1 <= rank <= gallery_size:
        raise ValueError(f"rank 必须在 1 到 {gallery_size} 之间。")
    return {
        "recall@1": float(rank <= 1),
        "recall@5": float(rank <= 5),
        "recall@10": float(rank <= 10),
        "reciprocal_rank": 1.0 / rank,
        "ndcg": 1.0 / np.log2(rank + 1),
        "ndcg@10": (1.0 / np.log2(rank + 1)) if rank <= 10 else 0.0,
    }


def evaluate_rankings(
    records: list[dict[str, str]], rankings: list[list[dict[str, Any]]], image_manifest: list[str]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if len(records) != len(rankings):
        raise ValueError("Ground truth 与检索结果数量不一致。")
    rows = []
    metric_names = ("recall@1", "recall@5", "recall@10", "reciprocal_rank", "ndcg", "ndcg@10")
    for record, ranking in zip(records, rankings):
        target = record["image_path"]
        rank = next((i for i, result in enumerate(ranking, start=1) if result["path"].relative_to(ROOT).as_posix() == target), None)
        if rank is None:
            # The CLI asks for a complete ranking; silently truncated rankings are a bug.
            raise ValueError(f"完整排序中没有找到目标图片：{target}")
        metrics = calculate_metrics(rank, len(image_manifest))
        rows.append({**record, "rank": rank, **metrics, "top10": ranking[:10]})
    summary = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    return summary, rows
