#!/usr/bin/env python3
"""Evaluate CLIP image retrieval against a JSON ground-truth set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from clip_retrieval import (
    MODEL_PATH,
    ROOT,
    encode_image_gallery,
    evaluate_rankings,
    load_model,
    prepare_dataset,
    rank_queries,
    validate_ground_truth,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_reports(
    output_dir: Path,
    summary: dict[str, float],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {"metadata": metadata, "query_count": len(rows), "metrics": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "query_id", "query", "image_path", "rank", "recall@1", "recall@5",
        "recall@10", "reciprocal_rank", "ndcg", "ndcg@10", "top10",
    ]
    with (output_dir / "queries.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in columns if key != "top10"},
                "top10": json.dumps([
                    {"image_path": result["path"].relative_to(ROOT).as_posix(), "score": result["score"]}
                    for result in row["top10"]
                ], ensure_ascii=False),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/ground_truth.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    args = parser.parse_args()

    ground_truth_path = args.ground_truth.resolve()
    try:
        ground_truth_bytes = ground_truth_path.read_bytes()
        raw_records = json.loads(ground_truth_bytes)
    except FileNotFoundError as error:
        raise SystemExit(f"找不到 ground truth 文件：{ground_truth_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Ground truth JSON 格式错误：{error}") from error

    image_paths, image_manifest = prepare_dataset()
    records = validate_ground_truth(raw_records, image_manifest)
    device, processor, model = load_model()
    image_features = encode_image_gallery(image_paths, image_manifest, device, processor, model)
    rankings = rank_queries(
        [record["query"] for record in records], image_features, image_paths,
        device, processor, model, top_k=len(image_paths),
    )
    summary, rows = evaluate_rankings(records, rankings, image_manifest)

    gallery_facts = []
    for image_path, relative_path in zip(image_paths, image_manifest):
        stat = image_path.stat()
        gallery_facts.append((relative_path, stat.st_size, stat.st_mtime_ns))
    gallery_fingerprint = sha256_bytes(
        json.dumps(gallery_facts, ensure_ascii=False, separators=(",", ":")).encode()
    )
    model_path = Path(MODEL_PATH)
    if model_path.is_dir():
        model_facts = []
        for path in sorted(model_path.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                stat = path.stat()
                model_facts.append((path.relative_to(model_path).as_posix(), stat.st_size, stat.st_mtime_ns))
        model_identity = json.dumps(model_facts, separators=(",", ":"))
    else:
        model_identity = str(MODEL_PATH)
    model_fingerprint = sha256_bytes(
        (model_identity + ":" + str(model.config.projection_dim)).encode()
    )
    metadata = {
        "model": str(MODEL_PATH),
        "model_fingerprint": model_fingerprint,
        "device": str(device),
        "gallery_size": len(image_manifest),
        "gallery_fingerprint": gallery_fingerprint,
        "ground_truth": str(ground_truth_path),
        "ground_truth_fingerprint": sha256_bytes(ground_truth_bytes),
    }
    write_reports(args.output_dir, summary, rows, metadata)

    print(f"评测 query 数：{len(rows)}；图库图片数：{len(image_manifest)}")
    for name, value in summary.items():
        label = {"reciprocal_rank": "MRR"}.get(name, name.upper() if name.startswith("ndcg") else name.replace("recall", "Recall"))
        print(f"{label}: {value:.4f} ({value:.1%})")
    print(f"汇总：{args.output_dir / 'summary.json'}")
    print(f"逐 query：{args.output_dir / 'queries.csv'}")


if __name__ == "__main__":
    main()
