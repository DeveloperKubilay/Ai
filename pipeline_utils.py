import hashlib
import json
import math
import os
from typing import Any


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config(path: str = "settings.json") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_run_id(config: dict[str, Any], train_path: str = "train.jsonl") -> str:
    preprocessing_cfg = config.get("preprocessing", {})
    payload = {
        "format_version": 1,
        "base_model": config["model"]["base_model"],
        "train_sha256": sha256_file(train_path),
        "max_length_cap": preprocessing_cfg.get("max_length_cap"),
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def get_prepared_paths(config: dict[str, Any], train_path: str = "train.jsonl") -> dict[str, str]:
    run_id = build_run_id(config, train_path)
    run_key = run_id[:12]
    prepared_root = config.get("preprocessing", {}).get("prepared_root", "./prepared-datasets")
    run_dir = os.path.join(prepared_root, run_key)
    return {
        "run_id": run_id,
        "run_key": run_key,
        "root_dir": prepared_root,
        "run_dir": run_dir,
        "parts_dir": os.path.join(run_dir, "parts"),
        "manifest_path": os.path.join(run_dir, "manifest.json"),
        "dataset_dir": os.path.join(run_dir, "dataset"),
    }


def get_checkpoint_dir(config: dict[str, Any], run_key: str) -> str:
    checkpoint_root = config.get("checkpointing", {}).get("root_dir", "./checkpoints")
    return os.path.join(checkpoint_root, run_key)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def compute_context_window(lengths: list[int], max_length_cap: int | None = None) -> int:
    max_tokens = max(lengths)
    candidates = [256, 512, 1024, 2048, 4096, 8192]

    if max_length_cap is not None:
        candidates = [value for value in candidates if value <= max_length_cap]
        if not candidates:
            candidates = [max_length_cap]

    for ctx in candidates:
        if max_tokens <= ctx:
            return ctx

    return candidates[-1]


def optimizer_steps_per_epoch(sample_count: int, batch_size: int, grad_accum: int) -> int:
    micro_batches = math.ceil(sample_count / batch_size)
    return max(1, math.ceil(micro_batches / grad_accum))


def build_training_profile(sample_count: int, token_lengths: list[int], training_cfg: dict[str, Any], preprocessing_cfg: dict[str, Any]) -> dict[str, Any]:
    auto_cfg = training_cfg.get("auto", {})
    auto_enabled = auto_cfg.get("enabled", True)

    base_batch = max(1, int(training_cfg["batch_size"]))
    base_grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 4)))
    base_epochs = max(1, int(training_cfg["num_epochs"]))
    base_lr = float(training_cfg["learning_rate"])
    base_r = max(4, int(training_cfg["lora_r"]))
    base_alpha = max(int(training_cfg["lora_alpha"]), base_r * 2)
    max_length_cap = preprocessing_cfg.get("max_length_cap")

    profile = {
        "auto_enabled": auto_enabled,
        "batch_size": base_batch,
        "grad_accum": base_grad_accum,
        "epochs": base_epochs,
        "learning_rate": base_lr,
        "lora_r": base_r,
        "lora_alpha": base_alpha,
        "context_window": compute_context_window(token_lengths, max_length_cap=max_length_cap),
        "target_updates": optimizer_steps_per_epoch(sample_count, base_batch, base_grad_accum) * base_epochs,
    }

    if not auto_enabled:
        return profile

    reference_examples = max(1, int(auto_cfg.get("reference_examples", 48)))
    min_epochs = max(1, int(auto_cfg.get("min_epochs", 4)))
    max_epochs = max(min_epochs, int(auto_cfg.get("max_epochs", max(base_epochs * 2, 60))))

    if sample_count < 32:
        grad_accum = 1
    elif sample_count < 128:
        grad_accum = 2
    elif sample_count < 512:
        grad_accum = 4
    elif sample_count < 4096:
        grad_accum = 8
    else:
        grad_accum = 16

    if sample_count < 64:
        lr_cap = 2e-4
    elif sample_count < 256:
        lr_cap = 3e-4
    elif sample_count < 1024:
        lr_cap = 5e-4
    else:
        lr_cap = base_lr

    if sample_count < 2048:
        lora_r = base_r
    elif sample_count < 8192:
        lora_r = max(base_r, 24)
    else:
        lora_r = max(base_r, 32)

    reference_updates = optimizer_steps_per_epoch(reference_examples, base_batch, base_grad_accum) * base_epochs
    target_updates = max(40, reference_updates)
    steps_per_epoch = optimizer_steps_per_epoch(sample_count, base_batch, grad_accum)
    epochs = math.ceil(target_updates / steps_per_epoch)
    epochs = max(min_epochs, min(max_epochs, epochs))

    profile.update(
        {
            "batch_size": base_batch,
            "grad_accum": grad_accum,
            "epochs": epochs,
            "learning_rate": min(base_lr, lr_cap),
            "lora_r": lora_r,
            "lora_alpha": max(base_alpha, lora_r * 2),
            "target_updates": target_updates,
        }
    )
    return profile


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_names = [
        name for name in os.listdir(checkpoint_dir) if name.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, name))
    ]
    if not checkpoint_names:
        return None

    latest_name = max(checkpoint_names, key=lambda value: int(value.split("-")[1]))
    return os.path.join(checkpoint_dir, latest_name)


def compute_save_steps(steps_per_epoch: int, checkpoint_cfg: dict[str, Any]) -> int:
    requested = int(checkpoint_cfg.get("save_steps", 50))
    min_save_steps = max(1, int(checkpoint_cfg.get("min_save_steps", 1)))
    return max(min_save_steps, min(requested, steps_per_epoch))


def compute_logging_steps(steps_per_epoch: int) -> int:
    return max(1, min(5, steps_per_epoch))
