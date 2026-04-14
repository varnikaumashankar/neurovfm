import argparse
import datetime
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovfm.models import AggregateThenClassify, ClassifyThenAggregate, MLP

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


EMBED_SUFFIX = "_encoder_embeddings.pt"


def parse_hidden_dims(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part) for part in raw.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_subject_id(subject_id: str, strip_subject_prefix: str) -> str:
    subject_id = str(subject_id).strip()
    if strip_subject_prefix and subject_id.startswith(strip_subject_prefix):
        return subject_id[len(strip_subject_prefix) :]
    return subject_id


def extract_subject_id(path: Path, strip_subject_prefix: str) -> str:
    name = path.name
    if name.endswith(EMBED_SUFFIX):
        name = name[: -len(EMBED_SUFFIX)]
    else:
        name = path.stem
    return canonicalize_subject_id(name, strip_subject_prefix)


def load_embedding_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "embedding" in data:
            data = data["embedding"]
        else:
            raise ValueError(f"Unsupported embedding dict format in {path}")
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}, found {type(data)!r}")
    if data.ndim != 2:
        raise ValueError(f"Expected [N_tokens, D] tensor in {path}, found shape {tuple(data.shape)}")
    return data.float()


def scan_embedding_paths(embeddings_dir: Path, strip_subject_prefix: str) -> dict[str, Path]:
    subject_to_path: dict[str, Path] = {}
    duplicate_subjects = []
    embedding_paths = sorted(embeddings_dir.rglob(f"*{EMBED_SUFFIX}"))
    if not embedding_paths:
        return {}

    for path in embedding_paths:
        subject_id = extract_subject_id(path, strip_subject_prefix)
        if subject_id in subject_to_path:
            duplicate_subjects.append(subject_id)
            continue
        subject_to_path[subject_id] = path

    if duplicate_subjects:
        print(f"Warning: found duplicate embedding files for {len(duplicate_subjects)} subjects; keeping first match")

    return subject_to_path


def load_labels(labels_path: Path, subject_col: str, target_col: str, sep: Optional[str], strip_subject_prefix: str) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if labels_path.suffix.lower() == ".tsv" else ","

    df = pd.read_csv(labels_path, sep=sep)
    missing = [col for col in (subject_col, target_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {labels_path}: {missing}")

    df = df[[subject_col, target_col]].copy()
    df = df.dropna(subset=[subject_col, target_col])
    df[subject_col] = df[subject_col].astype(str)
    df["_subject_key"] = df[subject_col].map(lambda value: canonicalize_subject_id(value, strip_subject_prefix))
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df = df.drop_duplicates(subset=["_subject_key"], keep="first")
    return df


def build_examples(
    subject_to_path: dict[str, Path],
    labels_df: pd.DataFrame,
    target_col: str,
    warn_missing: bool = True,
) -> list[dict]:
    examples = []
    missing_embeddings = []

    for row in labels_df[["_subject_key", target_col]].to_dict("records"):
        subject_id = row["_subject_key"]
        emb_path = subject_to_path.get(subject_id)
        if emb_path is None:
            missing_embeddings.append(subject_id)
            continue
        examples.append(
            {
                "subject_id": subject_id,
                "target": float(row[target_col]),
                "emb_path": emb_path,
            }
        )

    if not examples:
        return []

    if missing_embeddings and warn_missing:
        print(f"Warning: missing embeddings for {len(missing_embeddings)} labeled subjects")

    return examples


def compute_feature_stats_from_examples(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    feature_sum = None
    feature_sq_sum = None
    count = 0

    for ex in examples:
        pooled = load_embedding_tensor(ex["emb_path"]).mean(dim=0)
        if feature_sum is None:
            feature_sum = torch.zeros_like(pooled)
            feature_sq_sum = torch.zeros_like(pooled)
        feature_sum += pooled
        feature_sq_sum += pooled * pooled
        count += 1

    if count == 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("Cannot compute feature stats with zero training examples")

    feature_mean = feature_sum / count
    variance = feature_sq_sum / count - feature_mean * feature_mean
    feature_std = variance.clamp_min(1e-6).sqrt()
    return feature_mean, feature_std


def segment_reduce_mean(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    if torch_scatter is not None:
        return torch_scatter.segment_csr(src=src, indptr=indptr.long(), reduce="mean")

    outputs = []
    for idx in range(indptr.numel() - 1):
        start = int(indptr[idx].item())
        end = int(indptr[idx + 1].item())
        outputs.append(src[start:end].mean(dim=0))
    return torch.stack(outputs, dim=0)


class EmbeddingBagDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        embedding = load_embedding_tensor(ex["emb_path"])
        return {
            "embedding": embedding,
            "target": torch.tensor(ex["target"], dtype=torch.float32),
            "subject_id": ex["subject_id"],
        }


def make_collate_fn(feature_mean: Optional[torch.Tensor], feature_std: Optional[torch.Tensor]):
    def collate(batch: list[dict]) -> dict:
        embeddings = []
        subject_ids = []
        targets = []
        lengths = []

        for item in batch:
            emb = item["embedding"]
            if feature_mean is not None and feature_std is not None:
                emb = (emb - feature_mean) / feature_std
            embeddings.append(emb)
            subject_ids.append(item["subject_id"])
            targets.append(item["target"])
            lengths.append(emb.shape[0])

        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int64)
        if lengths:
            cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)

        return {
            "embeddings": torch.cat(embeddings, dim=0),
            "targets": torch.stack(targets, dim=0),
            "subject_ids": subject_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(lengths),
        }

    return collate


class EmbeddingRegressionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pooler: str,
        hidden_dims: list[int],
        mil_hidden_dim: int,
        addmil_output_bias_scale: bool,
    ):
        super().__init__()
        self.pooler_name = pooler

        if pooler == "avgpool":
            self.pooler = None
            self.head = MLP(in_dim=input_dim, out_dim=1, hidden_dims=hidden_dims)
        elif pooler == "abmil":
            self.pooler = AggregateThenClassify(
                dim=input_dim,
                hidden_dim=mil_hidden_dim,
                W_out=1,
                use_gating=True,
                use_norm=True,
            )
            self.head = MLP(in_dim=input_dim, out_dim=1, hidden_dims=hidden_dims)
        elif pooler == "addmil":
            self.pooler = ClassifyThenAggregate(
                dim=input_dim,
                hidden_dim=mil_hidden_dim,
                W_out=1,
                mlp_hidden_dims=hidden_dims,
                mlp_out_dim=1,
                use_gating=True,
                use_norm=False,
                use_output_bias_scale=addmil_output_bias_scale,
            )
            self.head = None
        else:
            raise ValueError(f"Unsupported pooler: {pooler}")

    def forward(self, embeddings: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        if self.pooler_name == "avgpool":
            pooled = segment_reduce_mean(embeddings, cu_seqlens)
            return self.head(pooled).squeeze(-1)

        pooled = self.pooler(embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if self.pooler_name == "abmil":
            if pooled.ndim == 3:
                pooled = pooled.squeeze(1)
            return self.head(pooled).squeeze(-1)

        return pooled.squeeze(-1)

    def predict_with_details(
        self,
        embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        if self.pooler_name == "avgpool":
            return self.forward(embeddings, cu_seqlens, max_seqlen), None

        if self.pooler_name == "abmil":
            pooled, attention_weights = self.pooler(
                embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                return_attn_probs=True,
            )
            if pooled.ndim == 3:
                pooled = pooled.squeeze(1)
            preds = self.head(pooled).squeeze(-1)
            return preds, {"attention_weights": attention_weights}

        output, attention_weights, patch_logits = self.pooler(
            embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_logits=True,
        )
        return output.squeeze(-1), {
            "attention_weights": attention_weights,
            "patch_logits": patch_logits,
        }


def train_on_loader(model, loader, optimizer, criterion, device) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        targets = batch["targets"].to(device)
        cu_seqlens = batch["cu_seqlens"].to(device)
        max_seqlen = batch["max_seqlen"]

        optimizer.zero_grad(set_to_none=True)
        preds = model(embeddings, cu_seqlens, max_seqlen)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss, count


@torch.inference_mode()
def evaluate_on_loader(model, loader, criterion, device, collect_attention: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    count = 0
    all_targets = []
    all_preds = []
    all_subjects = []
    attention_records = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        targets = batch["targets"].to(device)
        cu_seqlens = batch["cu_seqlens"].to(device)
        max_seqlen = batch["max_seqlen"]

        if collect_attention:
            preds, details = model.predict_with_details(embeddings, cu_seqlens, max_seqlen)
        else:
            preds = model(embeddings, cu_seqlens, max_seqlen)
            details = None
        loss = criterion(preds, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

        targets_cpu = targets.cpu()
        preds_cpu = preds.cpu()
        all_targets.append(targets_cpu)
        all_preds.append(preds_cpu)
        all_subjects.extend(batch["subject_ids"])

        if details is not None:
            attention_weights = details["attention_weights"].detach().cpu()
            patch_logits = details.get("patch_logits")
            if patch_logits is not None:
                patch_logits = patch_logits.detach().cpu()

            cu_seqlens_cpu = cu_seqlens.cpu()
            for idx, subject_id in enumerate(batch["subject_ids"]):
                start = int(cu_seqlens_cpu[idx].item())
                end = int(cu_seqlens_cpu[idx + 1].item())
                attn_slice = attention_weights[start:end].squeeze(-1).clone()
                record = {
                    "subject_id": subject_id,
                    "target": float(targets_cpu[idx].item()),
                    "prediction": float(preds_cpu[idx].item()),
                    "num_tokens": end - start,
                    "attention_weights": attn_slice,
                }
                if attn_slice.numel() > 0:
                    top_index = int(torch.argmax(attn_slice).item())
                    record["top_token_index"] = top_index
                    record["top_attention_weight"] = float(attn_slice[top_index].item())
                if patch_logits is not None:
                    record["patch_logits"] = patch_logits[start:end].squeeze(-1).clone()
                attention_records.append(record)

    if count == 0:
        return {
            "loss_sum": 0.0,
            "count": 0,
            "targets": [],
            "predictions": [],
            "subject_ids": [],
            "attention_records": [],
        }

    return {
        "loss_sum": total_loss,
        "count": count,
        "targets": torch.cat(all_targets).numpy().tolist(),
        "predictions": torch.cat(all_preds).numpy().tolist(),
        "subject_ids": all_subjects,
        "attention_records": attention_records,
    }


def summarize_predictions(loss_sum: float, count: int, targets: list[float], predictions: list[float]) -> dict:
    if count == 0 or not targets:
        return {
            "loss": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    targets_np = np.asarray(targets, dtype=np.float32)
    preds_np = np.asarray(predictions, dtype=np.float32)
    mse = mean_squared_error(targets_np, preds_np)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(targets_np, preds_np)
    try:
        r2 = r2_score(targets_np, preds_np)
    except ValueError:
        r2 = float("nan")

    return {
        "loss": loss_sum / count,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_attention_artifacts(path: Path, pooler: str, records: list[dict]) -> None:
    torch.save(
        {
            "pooler": pooler,
            "num_subjects": len(records),
            "records": records,
        },
        path,
    )


def save_attention_summary(path: Path, records: list[dict]) -> None:
    if not records:
        pd.DataFrame(
            columns=[
                "subject_id",
                "target",
                "prediction",
                "num_tokens",
                "top_token_index",
                "top_attention_weight",
            ]
        ).to_csv(path, index=False)
        return

    pd.DataFrame(
        [
            {
                "subject_id": record["subject_id"],
                "target": record["target"],
                "prediction": record["prediction"],
                "num_tokens": record["num_tokens"],
                "top_token_index": record.get("top_token_index"),
                "top_attention_weight": record.get("top_attention_weight"),
            }
            for record in records
        ]
    ).to_csv(path, index=False)


def maybe_save_attention_outputs(args, records: list[dict]) -> None:
    if args.pooler not in {"abmil", "addmil"}:
        return
    save_attention_artifacts(args.output_dir / "val_attention_details.pt", args.pooler, records)
    save_attention_summary(args.output_dir / "val_attention_summary.csv", records)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def download_remote_batch(args, batch_id: int) -> Path:
    raise NotImplementedError("download_remote_batch now expects a remote batch spec")


def cleanup_local_batch(local_batch_dir: Path) -> None:
    shutil.rmtree(local_batch_dir, ignore_errors=True)


def get_batch_ids(args) -> list[int]:
    return list(range(args.batch_start, args.batch_end + 1))


def run_rclone_lsf(remote_path: str) -> list[str]:
    result = subprocess.run(
        ["rclone", "lsf", remote_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_remote_batch_specs(args) -> list[dict]:
    remote_root = f"{args.rclone_remote}:{args.rclone_base_path}/{args.remote_embedding_subdir}"
    entries = run_rclone_lsf(remote_root)
    dir_entries = [entry.rstrip("/") for entry in entries if entry.endswith("/")]
    file_entries = [entry for entry in entries if entry.endswith(".pt")]

    if dir_entries:
        selected_dirs = []
        wanted = set(get_batch_ids(args))
        for entry in dir_entries:
            if entry.startswith("batch_"):
                try:
                    batch_id = int(entry.split("_", 1)[1])
                except ValueError:
                    continue
                if batch_id in wanted:
                    selected_dirs.append(
                        {
                            "kind": "dir",
                            "batch_id": batch_id,
                            "remote_path": f"{remote_root}/{entry}",
                        }
                    )
        if selected_dirs:
            return sorted(selected_dirs, key=lambda item: item["batch_id"])

    if not file_entries:
        raise FileNotFoundError(f"No remote embedding files found under {remote_root}")

    file_entries = sorted(file_entries)
    chunk_size = args.remote_chunk_size
    specs = []
    for idx in range(0, len(file_entries), chunk_size):
        logical_batch_id = idx // chunk_size + 1
        if logical_batch_id < args.batch_start or logical_batch_id > args.batch_end:
            continue
        specs.append(
            {
                "kind": "files",
                "batch_id": logical_batch_id,
                "remote_root": remote_root,
                "files": file_entries[idx : idx + chunk_size],
            }
        )
    return specs


def save_remote_batch_manifest(path: Path, batch_specs: list[dict]) -> None:
    if not batch_specs:
        save_json(path, {"layout": "empty", "batches": []})
        return

    if all(spec["kind"] == "dir" for spec in batch_specs):
        save_json(
            path,
            {
                "layout": "directory",
                "num_batches": len(batch_specs),
                "batches": [
                    {
                        "batch_id": spec["batch_id"],
                        "remote_path": spec["remote_path"],
                    }
                    for spec in batch_specs
                ],
            },
        )
        return

    all_files = [relpath for spec in batch_specs for relpath in spec["files"]]
    save_json(
        path,
        {
            "layout": "flat_files",
            "num_batches": len(batch_specs),
            "total_files": len(all_files),
            "batches": [
                {
                    "batch_id": spec["batch_id"],
                    "remote_root": spec["remote_root"],
                    "num_files": len(spec["files"]),
                    "first_file": spec["files"][0] if spec["files"] else None,
                    "last_file": spec["files"][-1] if spec["files"] else None,
                    "files": spec["files"],
                }
                for spec in batch_specs
            ],
        },
    )


def download_remote_batch(args, batch_spec: dict, force: bool = True) -> Path:
    batch_id = batch_spec["batch_id"]
    local_batch_dir = args.local_cache_dir / f"batch_{batch_id}"
    if not force and local_batch_dir.exists() and any(local_batch_dir.glob("*.pt")):
        return local_batch_dir
    shutil.rmtree(local_batch_dir, ignore_errors=True)
    local_batch_dir.mkdir(parents=True, exist_ok=True)

    if batch_spec["kind"] == "dir":
        remote_path = batch_spec["remote_path"]
        cmd = ["rclone", "copy", remote_path, str(local_batch_dir), "--include", "*.pt"]
        print(f"Downloading remote embeddings batch {batch_id} from {remote_path}")
        subprocess.run(cmd, check=True)
        return local_batch_dir

    remote_root = batch_spec["remote_root"]
    print(
        f"Downloading logical remote batch {batch_id} "
        f"with {len(batch_spec['files'])} embedding files from {remote_root}"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as handle:
        for relpath in batch_spec["files"]:
            handle.write(relpath)
            handle.write("\n")
        files_from_path = handle.name
    try:
        cmd = ["rclone", "copy", remote_root, str(local_batch_dir), "--files-from", files_from_path]
        subprocess.run(cmd, check=True)
    finally:
        Path(files_from_path).unlink(missing_ok=True)
    return local_batch_dir


def resolve_input_dim(args, batch_specs: list[dict]) -> int:
    if not args.use_remote_batches:
        if args.embeddings_dir is None:
            raise ValueError("--embeddings_dir is required when not using remote batches")
        subject_to_path = scan_embedding_paths(args.embeddings_dir, args.strip_subject_prefix)
        if not subject_to_path:
            raise FileNotFoundError(f"No embedding files found under {args.embeddings_dir}")
        first_path = next(iter(subject_to_path.values()))
        return load_embedding_tensor(first_path).shape[1]

    for batch_spec in batch_specs:
        local_batch_dir = download_remote_batch(args, batch_spec, force=False)
        subject_to_path = scan_embedding_paths(local_batch_dir, args.strip_subject_prefix)
        if subject_to_path:
            first_path = next(iter(subject_to_path.values()))
            return load_embedding_tensor(first_path).shape[1]

    raise FileNotFoundError("No remote embedding files found across configured batch range")


def compute_feature_stats_remote(args, train_labels_df: pd.DataFrame, batch_specs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    feature_sum = None
    feature_sq_sum = None
    count = 0

    for batch_spec in batch_specs:
        local_batch_dir = download_remote_batch(args, batch_spec, force=False)
        subject_to_path = scan_embedding_paths(local_batch_dir, args.strip_subject_prefix)
        train_examples = build_examples(subject_to_path, train_labels_df, args.target_col, warn_missing=False)
        for ex in train_examples:
            pooled = load_embedding_tensor(ex["emb_path"]).mean(dim=0)
            if feature_sum is None:
                feature_sum = torch.zeros_like(pooled)
                feature_sq_sum = torch.zeros_like(pooled)
            feature_sum += pooled
            feature_sq_sum += pooled * pooled
            count += 1

    if count == 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("No train examples found while computing remote feature statistics")

    feature_mean = feature_sum / count
    variance = feature_sq_sum / count - feature_mean * feature_mean
    feature_std = variance.clamp_min(1e-6).sqrt()
    return feature_mean, feature_std


def collect_local_attention_records(args, model, criterion, device, val_labels_df, feature_mean, feature_std) -> list[dict]:
    if args.pooler not in {"abmil", "addmil"}:
        return []

    subject_to_path = scan_embedding_paths(args.embeddings_dir, args.strip_subject_prefix)
    val_examples = build_examples(subject_to_path, val_labels_df, args.target_col, warn_missing=True)
    if not val_examples:
        return []

    val_loader = DataLoader(
        EmbeddingBagDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(feature_mean, feature_std),
    )
    payload = evaluate_on_loader(model, val_loader, criterion, device, collect_attention=True)
    return payload["attention_records"]


def collect_remote_attention_records(
    args,
    model,
    criterion,
    device,
    val_labels_df,
    feature_mean,
    feature_std,
    local_batch_dirs,
) -> list[dict]:
    if args.pooler not in {"abmil", "addmil"}:
        return []

    records = []
    for local_batch_dir in local_batch_dirs:
        subject_to_path = scan_embedding_paths(local_batch_dir, args.strip_subject_prefix)
        val_examples = build_examples(subject_to_path, val_labels_df, args.target_col, warn_missing=False)
        if not val_examples:
            continue
        val_loader = DataLoader(
            EmbeddingBagDataset(val_examples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=make_collate_fn(feature_mean, feature_std),
        )
        payload = evaluate_on_loader(model, val_loader, criterion, device, collect_attention=True)
        records.extend(payload["attention_records"])

    return records


def run_local_training(args, model, optimizer, criterion, device, train_labels_df, val_labels_df, feature_mean, feature_std):
    subject_to_path = scan_embedding_paths(args.embeddings_dir, args.strip_subject_prefix)
    if not subject_to_path:
        raise FileNotFoundError(f"No embedding files found under {args.embeddings_dir}")

    train_examples = build_examples(subject_to_path, train_labels_df, args.target_col, warn_missing=True)
    val_examples = build_examples(subject_to_path, val_labels_df, args.target_col, warn_missing=True)
    if not train_examples:
        raise ValueError("No training examples overlap with local embeddings")

    train_loader = DataLoader(
        EmbeddingBagDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(feature_mean, feature_std),
    )
    val_loader = DataLoader(
        EmbeddingBagDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(feature_mean, feature_std),
    )

    history = []
    best_val_rmse = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss_sum, train_count = train_on_loader(model, train_loader, optimizer, criterion, device)
        val_raw = evaluate_on_loader(model, val_loader, criterion, device)
        train_summary = summarize_predictions(train_loss_sum, train_count, [], [])
        val_summary = summarize_predictions(val_raw["loss_sum"], val_raw["count"], val_raw["targets"], val_raw["predictions"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_summary["loss"],
            "val_rmse": val_summary["rmse"],
            "val_mae": val_summary["mae"],
            "val_r2": val_summary["r2"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:03d} "
            f"pooler={args.pooler} "
            f"train_loss={epoch_metrics['train_loss']:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"val_rmse={val_summary['rmse']:.6f} "
            f"val_mae={val_summary['mae']:.6f} "
            f"val_r2={val_summary['r2']:.6f}"
        )

        if val_summary["rmse"] < best_val_rmse:
            best_val_rmse = val_summary["rmse"]
            best_epoch = epoch
            save_checkpoint(args, model, feature_mean, feature_std, best_val_rmse, epoch)
            save_val_predictions(args.output_dir / "val_predictions.csv", val_raw)
            maybe_save_attention_outputs(
                args,
                collect_local_attention_records(
                    args,
                    model,
                    criterion,
                    device,
                    val_labels_df,
                    feature_mean,
                    feature_std,
                ),
            )

    return history, best_epoch, best_val_rmse, len(train_examples), len(val_examples)


def run_remote_training(args, model, optimizer, criterion, device, train_labels_df, val_labels_df, feature_mean, feature_std, batch_specs):
    print(f"Pre-downloading {len(batch_specs)} remote embedding batches...")
    local_batch_dirs = [download_remote_batch(args, batch_spec, force=False) for batch_spec in batch_specs]
    print("All batches cached locally. Starting training loop.")

    history = []
    best_val_rmse = float("inf")
    best_epoch = -1

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss_sum = 0.0
            train_count = 0
            val_loss_sum = 0.0
            val_count = 0
            val_targets = []
            val_predictions = []
            val_subject_ids = []
            train_seen = 0
            val_seen = 0

            for local_batch_dir in local_batch_dirs:
                subject_to_path = scan_embedding_paths(local_batch_dir, args.strip_subject_prefix)
                train_examples = build_examples(subject_to_path, train_labels_df, args.target_col, warn_missing=False)
                val_examples = build_examples(subject_to_path, val_labels_df, args.target_col, warn_missing=False)

                if train_examples:
                    train_loader = DataLoader(
                        EmbeddingBagDataset(train_examples),
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=make_collate_fn(feature_mean, feature_std),
                    )
                    batch_loss_sum, batch_count = train_on_loader(model, train_loader, optimizer, criterion, device)
                    train_loss_sum += batch_loss_sum
                    train_count += batch_count
                    train_seen += len(train_examples)

                if val_examples:
                    val_loader = DataLoader(
                        EmbeddingBagDataset(val_examples),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=make_collate_fn(feature_mean, feature_std),
                    )
                    batch_val = evaluate_on_loader(model, val_loader, criterion, device)
                    val_loss_sum += batch_val["loss_sum"]
                    val_count += batch_val["count"]
                    val_targets.extend(batch_val["targets"])
                    val_predictions.extend(batch_val["predictions"])
                    val_subject_ids.extend(batch_val["subject_ids"])
                    val_seen += len(val_examples)

            if train_count == 0:
                raise ValueError("No training examples were found across remote batches")

            val_summary = summarize_predictions(val_loss_sum, val_count, val_targets, val_predictions)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "val_loss": val_summary["loss"],
                "val_rmse": val_summary["rmse"],
                "val_mae": val_summary["mae"],
                "val_r2": val_summary["r2"],
                "train_examples_seen": train_seen,
                "val_examples_seen": val_seen,
            }
            history.append(epoch_metrics)
            print(
                f"epoch={epoch:03d} "
                f"pooler={args.pooler} "
                f"train_loss={epoch_metrics['train_loss']:.6f} "
                f"val_loss={val_summary['loss']:.6f} "
                f"val_rmse={val_summary['rmse']:.6f} "
                f"val_mae={val_summary['mae']:.6f} "
                f"val_r2={val_summary['r2']:.6f} "
                f"train_seen={train_seen} "
                f"val_seen={val_seen}"
            )

            if val_summary["rmse"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]
                best_epoch = epoch
                save_checkpoint(args, model, feature_mean, feature_std, best_val_rmse, epoch)
                save_val_predictions(
                    args.output_dir / "val_predictions.csv",
                    {
                        "subject_ids": val_subject_ids,
                        "targets": val_targets,
                        "predictions": val_predictions,
                    },
                )
                maybe_save_attention_outputs(
                    args,
                    collect_remote_attention_records(
                        args,
                        model,
                        criterion,
                        device,
                        val_labels_df,
                        feature_mean,
                        feature_std,
                        local_batch_dirs,
                    ),
                )
    finally:
        print("Cleaning up local embedding cache...")
        for local_dir in local_batch_dirs:
            cleanup_local_batch(local_dir)

    return history, best_epoch, best_val_rmse


def save_checkpoint(args, model, feature_mean, feature_std, best_val_rmse, epoch):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pooler": args.pooler,
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "mil_hidden_dim": args.mil_hidden_dim,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "subject_col": args.subject_col,
        "target_col": args.target_col,
        "strip_subject_prefix": args.strip_subject_prefix,
        "best_val_rmse": best_val_rmse,
        "epoch": epoch,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")


def save_val_predictions(path: Path, payload: dict):
    pd.DataFrame(
        {
            "subject_id": payload["subject_ids"],
            "target": payload["targets"],
            "prediction": payload["predictions"],
        }
    ).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train a configurable pooled regression head on saved NeuroVFM embeddings")
    parser.add_argument("--embeddings_dir", type=Path, default=None, help="Local root directory containing batch_*/.../*_encoder_embeddings.pt")
    parser.add_argument("--labels_path", type=Path, required=True, help="CSV/TSV with subject id and regression target")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save checkpoints and metrics")
    parser.add_argument("--subject_col", type=str, default="participant_id", help="Subject id column in labels table")
    parser.add_argument("--target_col", type=str, default="age", help="Regression target column in labels table")
    parser.add_argument("--sep", type=str, default=None, help="Override label file separator, e.g. ',' or '\\t'")
    parser.add_argument("--strip_subject_prefix", type=str, default="sub-", help="Prefix stripped from both label ids and embedding ids before matching")
    parser.add_argument("--pooler", type=str, choices=["avgpool", "abmil", "addmil"], default="avgpool")
    parser.add_argument("--hidden_dims", type=str, default="256,64", help="Comma-separated hidden dims for the regression MLP; reused by addmil's internal MLP")
    parser.add_argument("--mil_hidden_dim", type=int, default=256, help="Hidden dimension for ABMIL/AddMIL attention scoring")
    parser.add_argument("--disable_feature_norm", action="store_true", help="Skip train-set feature normalization")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of studies per optimizer step")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--disable_addmil_output_bias_scale", action="store_true")
    parser.add_argument("--use_remote_batches", action="store_true", help="Stream embedding batches from remote storage with rclone")
    parser.add_argument("--rclone_remote", type=str, default="chyhsu")
    parser.add_argument("--rclone_base_path", type=str, default="neurovfm")
    parser.add_argument("--remote_embedding_subdir", type=str, default="train_embeddings")
    parser.add_argument("--local_cache_dir", type=Path, default=Path("/home/chyhsu/Documents/training_embedding_cache"))
    parser.add_argument("--batch_start", type=int, default=1)
    parser.add_argument("--batch_end", type=int, default=33)
    parser.add_argument("--remote_chunk_size", type=int, default=100, help="Logical remote batch size when remote embeddings are stored flat")
    args = parser.parse_args()

    args.run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = args.output_dir / args.run_ts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.use_remote_batches and shutil.which("rclone") is None:
        raise RuntimeError("rclone is required when using --use_remote_batches")
    if not args.use_remote_batches and args.embeddings_dir is None:
        raise ValueError("--embeddings_dir is required unless --use_remote_batches is set")

    device = args.device
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    labels_df = load_labels(args.labels_path, args.subject_col, args.target_col, args.sep, args.strip_subject_prefix)
    train_labels_df, val_labels_df = train_test_split(
        labels_df,
        test_size=args.val_fraction,
        random_state=args.seed,
    )

    batch_specs = build_remote_batch_specs(args) if args.use_remote_batches else []
    if args.use_remote_batches:
        save_remote_batch_manifest(args.output_dir / "remote_batch_manifest.json", batch_specs)
        if batch_specs and all(spec["kind"] == "files" for spec in batch_specs):
            total_files = sum(len(spec["files"]) for spec in batch_specs)
            print(
                f"Using flat remote embedding layout: {total_files} files "
                f"split into {len(batch_specs)} logical batches "
                f"of up to {args.remote_chunk_size} files with no overlap"
            )
    input_dim = resolve_input_dim(args, batch_specs)

    feature_mean = None
    feature_std = None
    if not args.disable_feature_norm:
        if args.use_remote_batches:
            feature_mean, feature_std = compute_feature_stats_remote(args, train_labels_df, batch_specs)
        else:
            subject_to_path = scan_embedding_paths(args.embeddings_dir, args.strip_subject_prefix)
            train_examples = build_examples(subject_to_path, train_labels_df, args.target_col, warn_missing=True)
            if not train_examples:
                raise ValueError("No training examples overlap with local embeddings")
            feature_mean, feature_std = compute_feature_stats_from_examples(train_examples)

    model = EmbeddingRegressionModel(
        input_dim=input_dim,
        pooler=args.pooler,
        hidden_dims=hidden_dims,
        mil_hidden_dim=args.mil_hidden_dim,
        addmil_output_bias_scale=not args.disable_addmil_output_bias_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if args.use_remote_batches:
        history, best_epoch, best_val_rmse = run_remote_training(
            args,
            model,
            optimizer,
            criterion,
            device,
            train_labels_df,
            val_labels_df,
            feature_mean,
            feature_std,
            batch_specs,
        )
        num_train = len(train_labels_df)
        num_val = len(val_labels_df)
    else:
        history, best_epoch, best_val_rmse, num_train, num_val = run_local_training(
            args,
            model,
            optimizer,
            criterion,
            device,
            train_labels_df,
            val_labels_df,
            feature_mean,
            feature_std,
        )

    config_payload = {
        "embeddings_dir": str(args.embeddings_dir) if args.embeddings_dir is not None else None,
        "labels_path": str(args.labels_path),
        "output_dir": str(args.output_dir),
        "subject_col": args.subject_col,
        "target_col": args.target_col,
        "strip_subject_prefix": args.strip_subject_prefix,
        "pooler": args.pooler,
        "hidden_dims": hidden_dims,
        "mil_hidden_dim": args.mil_hidden_dim,
        "feature_norm_enabled": not args.disable_feature_norm,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": device,
        "use_remote_batches": args.use_remote_batches,
        "rclone_remote": args.rclone_remote,
        "rclone_base_path": args.rclone_base_path,
        "remote_embedding_subdir": args.remote_embedding_subdir,
        "local_cache_dir": str(args.local_cache_dir),
        "batch_start": args.batch_start,
        "batch_end": args.batch_end,
        "remote_chunk_size": args.remote_chunk_size,
        "num_train_label_rows": num_train,
        "num_val_label_rows": num_val,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }
    save_json(args.output_dir / "train_config.json", config_payload)
    save_json(args.output_dir / "history.json", {"epochs": history})

    print(f"Saved best model to {args.output_dir / 'best_model.pt'}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val RMSE: {best_val_rmse:.6f}")


if __name__ == "__main__":
    main()
