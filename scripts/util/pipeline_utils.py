import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SYSTEM_PROMPT = (
    "Sen kisa, dogal, yardimci ve durust Turkce cevaplar veren bir asistansin. "
    "Emin olmadiginda bunu acikca soyle ve uydurma bilgi verme."
)
ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
MESSAGE_ROLE_ALIASES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_project_path(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def resolve_model_reference(path_or_id: str) -> str:
    candidate = Path(path_or_id)
    if candidate.is_absolute():
        return str(candidate)
    if path_or_id.startswith(".") or path_or_id.startswith(".."):
        return resolve_project_path(path_or_id)

    local_candidate = PROJECT_ROOT / candidate
    if local_candidate.exists():
        return str(local_candidate.resolve())
    return path_or_id


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path or project_path("data", "settings.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    file_path = resolve_project_path(path)
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_run_id(config: dict[str, Any], train_path: str | None = None) -> str:
    train_path = resolve_project_path(train_path or project_path("data", "train.jsonl"))
    preprocessing_cfg = config.get("preprocessing", {})
    payload = {
        "format_version": 3,
        "base_model": config["model"]["base_model"],
        "train_sha256": sha256_file(train_path),
        "max_length_cap": preprocessing_cfg.get("max_length_cap"),
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def get_prepared_paths(config: dict[str, Any], train_path: str | None = None) -> dict[str, str]:
    train_path = resolve_project_path(train_path or project_path("data", "train.jsonl"))
    run_id = build_run_id(config, train_path)
    run_key = run_id[:12]
    prepared_root = resolve_project_path(config.get("preprocessing", {}).get("prepared_root", "./prepared-datasets"))
    run_dir = os.path.join(prepared_root, run_key)
    return {
        "run_id": run_id,
        "run_key": run_key,
        "root_dir": prepared_root,
        "run_dir": run_dir,
        "parts_dir": os.path.join(run_dir, "parts"),
        "manifest_path": os.path.join(run_dir, "manifest.json"),
        "dataset_dir": os.path.join(run_dir, "dataset"),
        "train_path": train_path,
    }


def get_checkpoint_dir(config: dict[str, Any], run_key: str) -> str:
    checkpoint_root = resolve_project_path(config.get("checkpointing", {}).get("root_dir", "./checkpoints"))
    return os.path.join(checkpoint_root, run_key)


def read_json(path: str, default: Any = None) -> Any:
    json_path = resolve_project_path(path)
    if not os.path.exists(json_path):
        return default
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, value: Any) -> None:
    json_path = resolve_project_path(path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def get_system_prompt(config: dict[str, Any] | None = None) -> str:
    if not config:
        return DEFAULT_SYSTEM_PROMPT

    return config.get("model", {}).get("system_prompt", DEFAULT_SYSTEM_PROMPT)


def normalize_message_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if not role:
        raise ValueError("Mesaj rol bilgisi bos.")

    normalized = MESSAGE_ROLE_ALIASES.get(role)
    if not normalized or normalized not in ALLOWED_MESSAGE_ROLES:
        raise ValueError(f"Desteklenmeyen mesaj rolu: {value}")
    return normalized


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError("Her mesaj dict olmalidir.")

    role = normalize_message_role(message.get("role", message.get("type")))
    content = message.get("content", message.get("response", message.get("text", "")))
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    content = str(content or "").strip()
    if not content:
        raise ValueError("Mesaj icerigi bos olamaz.")

    return {
        "role": role,
        "content": content,
    }


def normalize_messages(
    messages: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    inject_system: bool = True,
    require_assistant: bool = False,
    require_final_assistant: bool = False,
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages alani bos veya gecersiz.")

    normalized = [normalize_message(message) for message in messages]
    has_system_message = any(message["role"] == "system" for message in normalized)

    if inject_system and not has_system_message:
        normalized = [{"role": "system", "content": system_prompt}] + normalized

    if require_assistant and not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("Egitim orneginde en az bir assistant mesaji olmalidir.")

    if require_final_assistant and normalized[-1]["role"] != "assistant":
        raise ValueError("Egitim ornegi assistant mesaji ile bitmelidir.")

    return normalized


def build_single_turn_messages(
    question: str,
    answer: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(question).strip()},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer).strip()})
    return messages


def build_fallback_chat_text(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    rendered_parts = []
    for message in messages:
        rendered_parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")

    if add_generation_prompt:
        rendered_parts.append("<|im_start|>assistant\n")

    return "\n".join(rendered_parts)


def render_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any | None = None,
    tokenizer_source: str | None = None,
    add_generation_prompt: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    normalized = normalize_messages(messages, system_prompt=system_prompt)

    if tokenizer is None and tokenizer_source:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(resolve_model_reference(tokenizer_source))

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    return build_fallback_chat_text(normalized, add_generation_prompt=add_generation_prompt)


def render_token_ids(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[int]:
    normalized = normalize_messages(messages, system_prompt=system_prompt)

    if getattr(tokenizer, "chat_template", None):
        return list(
            tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        )

    rendered_text = build_fallback_chat_text(normalized, add_generation_prompt=add_generation_prompt)
    return list(tokenizer(rendered_text, add_special_tokens=False)["input_ids"])


def tokenize_messages_for_training(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    eos_token_id: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    normalized = normalize_messages(
        messages,
        system_prompt=system_prompt,
        require_assistant=True,
        require_final_assistant=True,
    )
    input_ids = render_token_ids(normalized, tokenizer, add_generation_prompt=False, system_prompt=system_prompt)
    labels = [-100] * len(input_ids)

    for index, message in enumerate(normalized):
        if message["role"] != "assistant":
            continue

        prompt_ids = render_token_ids(
            normalized[:index],
            tokenizer,
            add_generation_prompt=True,
            system_prompt=system_prompt,
        )
        full_turn_ids = render_token_ids(
            normalized[: index + 1],
            tokenizer,
            add_generation_prompt=False,
            system_prompt=system_prompt,
        )

        if len(full_turn_ids) <= len(prompt_ids) or full_turn_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Assistant token mask hesaplanamadi.")

        for position in range(len(prompt_ids), len(full_turn_ids)):
            labels[position] = full_turn_ids[position]

    if eos_token_id is not None and (not input_ids or input_ids[-1] != eos_token_id):
        input_ids.append(eos_token_id)
        labels.append(eos_token_id)

    return {
        "messages": normalized,
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
    }


def build_chat_text(
    question: str,
    answer: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tokenizer: Any | None = None,
    tokenizer_source: str | None = None,
) -> str:
    messages = build_single_turn_messages(question, answer, system_prompt=system_prompt)
    return render_messages(
        messages,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        add_generation_prompt=answer is None,
        system_prompt=system_prompt,
    )


def detect_runtime() -> dict[str, Any]:
    import torch

    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        return {
            "name": "tpu",
            "dtype": torch.bfloat16,
            "is_tpu": True,
            "is_cuda": False,
            "is_cpu": False,
        }
    except ImportError:
        pass

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return {
            "name": "cuda",
            "dtype": torch.float16,
            "is_tpu": False,
            "is_cuda": True,
            "is_cpu": False,
            "device_index": device_index,
            "device_name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "device_count": torch.cuda.device_count(),
        }

    return {
        "name": "cpu",
        "dtype": torch.float32,
        "is_tpu": False,
        "is_cuda": False,
        "is_cpu": True,
        "device_index": None,
        "device_name": "cpu",
        "total_memory_bytes": 0,
        "total_memory_gb": 0.0,
        "device_count": 1,
    }


def call_with_dtype_fallback(loader: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return loader(*args, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs or "unexpected keyword argument" not in str(exc):
            raise

        fallback_kwargs = dict(kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        print("Not: Bu transformers surumu `dtype` yerine `torch_dtype` bekliyor. Geri uyum modu kullaniliyor.")
        return loader(*args, **fallback_kwargs)


def build_compatible_init_kwargs(factory: Any, raw_kwargs: dict[str, Any], label: str) -> dict[str, Any]:
    supported_args = inspect.signature(factory).parameters
    compatible_kwargs = {}
    skipped_args = []

    for key, value in raw_kwargs.items():
        if key in supported_args:
            compatible_kwargs[key] = value
        else:
            skipped_args.append(key)

    if skipped_args:
        skipped_list = ", ".join(sorted(skipped_args))
        print(f"Not: Bu ortamda desteklenmeyen {label} argumanlari atlandi: {skipped_list}")

    return compatible_kwargs


def safe_torch_load(path: str) -> Any:
    import torch

    resolved_path = resolve_project_path(path)
    try:
        return torch.load(resolved_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(resolved_path, map_location="cpu")


def read_checkpoint_training_args(checkpoint_path: str) -> Any:
    args_path = os.path.join(resolve_project_path(checkpoint_path), "training_args.bin")
    if not os.path.exists(args_path):
        return None
    return safe_torch_load(args_path)


def build_checkpoint_resume_plan(
    checkpoint_path: str | None,
    profile: dict[str, Any],
    runtime: dict[str, Any],
    save_steps: int,
    logging_steps: int,
) -> dict[str, Any]:
    plan = {
        "checkpoint_path": checkpoint_path,
        "resume_trainer_state": False,
        "load_adapter": True,
        "reasons": [],
    }
    if not checkpoint_path:
        return plan

    checkpoint_path = resolve_project_path(checkpoint_path)

    adapter_config = read_json(os.path.join(checkpoint_path, "adapter_config.json"))
    if adapter_config is not None:
        if int(adapter_config.get("r", profile["lora_r"])) != int(profile["lora_r"]):
            plan["reasons"].append(f"lora_r: {adapter_config.get('r')} -> {profile['lora_r']}")
        if int(adapter_config.get("lora_alpha", profile["lora_alpha"])) != int(profile["lora_alpha"]):
            plan["reasons"].append(f"lora_alpha: {adapter_config.get('lora_alpha')} -> {profile['lora_alpha']}")
        if any(reason.startswith("lora_") for reason in plan["reasons"]):
            plan["load_adapter"] = False

    saved_args = read_checkpoint_training_args(checkpoint_path)
    if saved_args is None:
        plan["reasons"].append("training_args.bin bulunamadi")
        return plan

    comparisons = {
        "per_device_train_batch_size": profile["batch_size"],
        "gradient_accumulation_steps": profile["grad_accum"],
        "learning_rate": profile["learning_rate"],
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "fp16": runtime["is_cuda"],
        "bf16": runtime["is_tpu"],
    }

    for key, current_value in comparisons.items():
        if not hasattr(saved_args, key):
            continue

        saved_value = getattr(saved_args, key)
        if isinstance(current_value, float):
            is_same = math.isclose(float(saved_value), float(current_value), rel_tol=1e-9, abs_tol=1e-12)
        else:
            is_same = saved_value == current_value

        if not is_same:
            plan["reasons"].append(f"{key}: {saved_value} -> {current_value}")

    plan["resume_trainer_state"] = not plan["reasons"]
    return plan


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


def compute_adaptive_min_epochs(sample_count: int, configured_min_epochs: int) -> int:
    if sample_count >= 10000:
        return 1
    if sample_count >= 1024:
        return min(configured_min_epochs, 2)
    if sample_count >= 128:
        return min(configured_min_epochs, 3)
    return configured_min_epochs


def build_eval_split_plan(sample_count: int, evaluation_cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = evaluation_cfg.get("enabled", True)
    ratio = float(evaluation_cfg.get("ratio", 0.125))
    min_eval_examples = max(1, int(evaluation_cfg.get("min_examples", 2)))
    min_train_examples = max(1, int(evaluation_cfg.get("min_train_examples", 8)))
    patience = max(1, int(evaluation_cfg.get("patience", 2)))
    threshold = float(evaluation_cfg.get("threshold", 0.0))

    if not enabled or sample_count < (min_eval_examples + min_train_examples):
        return {
            "enabled": False,
            "eval_examples": 0,
            "train_examples": sample_count,
            "patience": patience,
            "threshold": threshold,
        }

    eval_examples = max(min_eval_examples, math.ceil(sample_count * ratio))
    eval_examples = min(eval_examples, sample_count - min_train_examples)
    eval_examples = max(1, eval_examples)

    return {
        "enabled": eval_examples > 0,
        "eval_examples": eval_examples,
        "train_examples": sample_count - eval_examples,
        "patience": patience,
        "threshold": threshold,
    }


def get_runtime_auto_value(
    auto_cfg: dict[str, Any], key: str, runtime_name: str, defaults: dict[str, int | float]
) -> int | float:
    value = auto_cfg.get(key)
    if isinstance(value, dict):
        if runtime_name in value:
            return value[runtime_name]
        if "default" in value:
            return value["default"]
    if runtime_name in defaults:
        return defaults[runtime_name]
    return defaults["default"]


def scale_cuda_capacity_targets(
    runtime: dict[str, Any],
    max_batch_size: int,
    target_device_tokens: int,
    target_step_tokens: int,
    max_grad_accum: int,
) -> tuple[int, int, int, int]:
    if not runtime.get("is_cuda"):
        return max_batch_size, target_device_tokens, target_step_tokens, max_grad_accum

    total_memory_gb = float(runtime.get("total_memory_gb", 0.0) or 0.0)
    if total_memory_gb >= 22:
        multiplier = 4
    elif total_memory_gb >= 14:
        multiplier = 3
    elif total_memory_gb >= 10:
        multiplier = 2
    else:
        multiplier = 1

    return (
        max(1, int(math.ceil(max_batch_size * multiplier))),
        max(1, int(target_device_tokens * multiplier)),
        max(1, int(target_step_tokens * multiplier)),
        max(1, int(math.ceil(max_grad_accum * max(1.0, multiplier / 2)))),
    )


def compute_auto_batch_profile(
    sample_count: int, context_window: int, runtime: dict[str, Any], auto_cfg: dict[str, Any]
) -> dict[str, int]:
    runtime_name = runtime["name"]
    max_batch_defaults = {
        "cpu": 1,
        "cuda": 8,
        "tpu": 32,
        "default": 4,
    }
    device_token_defaults = {
        "cpu": 1024,
        "cuda": 4096,
        "tpu": 16384,
        "default": 2048,
    }
    step_token_defaults = {
        "cpu": 1024,
        "cuda": 16384,
        "tpu": 65536,
        "default": 4096,
    }
    max_grad_defaults = {
        "cpu": 4,
        "cuda": 8,
        "tpu": 16,
        "default": 8,
    }

    max_batch_size = max(
        1,
        int(get_runtime_auto_value(auto_cfg, "max_batch_size", runtime_name, max_batch_defaults)),
    )
    target_device_tokens = max(
        1,
        int(get_runtime_auto_value(auto_cfg, "device_batch_tokens", runtime_name, device_token_defaults)),
    )
    target_step_tokens = max(
        1,
        int(get_runtime_auto_value(auto_cfg, "effective_step_tokens", runtime_name, step_token_defaults)),
    )
    max_grad_accum = max(
        1,
        int(get_runtime_auto_value(auto_cfg, "max_grad_accum", runtime_name, max_grad_defaults)),
    )
    max_batch_size, target_device_tokens, target_step_tokens, max_grad_accum = scale_cuda_capacity_targets(
        runtime,
        max_batch_size,
        target_device_tokens,
        target_step_tokens,
        max_grad_accum,
    )

    raw_batch_size = max(1, target_device_tokens // max(1, context_window))
    batch_size = min(max_batch_size, raw_batch_size)
    batch_size = min(max(1, sample_count), batch_size)

    micro_batches = max(1, math.ceil(sample_count / batch_size))
    grad_accum = max(1, math.ceil(target_step_tokens / max(1, batch_size * context_window)))
    grad_accum = min(grad_accum, max_grad_accum, micro_batches)

    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_batch_size": max_batch_size,
        "target_device_tokens": target_device_tokens,
        "target_step_tokens": target_step_tokens,
        "max_grad_accum": max_grad_accum,
    }


def compute_padding_multiple(runtime_name: str, auto_cfg: dict[str, Any]) -> int:
    defaults = {
        "cpu": 8,
        "cuda": 8,
        "tpu": 32,
        "default": 8,
    }
    return max(1, int(get_runtime_auto_value(auto_cfg, "pad_to_multiple_of", runtime_name, defaults)))


def min_steps_per_epoch_target(sample_count: int) -> int:
    if sample_count < 64:
        return 8
    if sample_count < 256:
        return 4
    return 1


def enforce_min_steps_per_epoch(sample_count: int, batch_size: int, grad_accum: int) -> tuple[int, int]:
    target_steps = min_steps_per_epoch_target(sample_count)
    current_steps = optimizer_steps_per_epoch(sample_count, batch_size, grad_accum)

    while current_steps < target_steps:
        if grad_accum > 1:
            grad_accum = max(1, grad_accum // 2)
        elif batch_size > 1:
            batch_size = max(1, batch_size // 2)
        else:
            break
        current_steps = optimizer_steps_per_epoch(sample_count, batch_size, grad_accum)

    return batch_size, grad_accum


def build_training_profile(
    sample_count: int,
    token_lengths: list[int],
    training_cfg: dict[str, Any],
    preprocessing_cfg: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auto_cfg = training_cfg.get("auto", {})
    auto_enabled = auto_cfg.get("enabled", True)
    runtime_name = (runtime or {}).get("name", "cpu")

    base_batch = max(1, int(training_cfg["batch_size"]))
    base_grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 4)))
    base_epochs = max(1, int(training_cfg["num_epochs"]))
    base_lr = float(training_cfg["learning_rate"])
    base_r = max(4, int(training_cfg["lora_r"]))
    base_alpha = max(int(training_cfg["lora_alpha"]), base_r * 2)
    max_length_cap = preprocessing_cfg.get("max_length_cap")
    context_window = compute_context_window(token_lengths, max_length_cap=max_length_cap)
    avg_tokens = max(1.0, sum(token_lengths) / max(1, sample_count))
    total_tokens = int(sum(token_lengths))
    pad_to_multiple_of = compute_padding_multiple(runtime_name, auto_cfg)

    profile = {
        "auto_enabled": auto_enabled,
        "runtime": runtime_name,
        "batch_size": base_batch,
        "grad_accum": base_grad_accum,
        "effective_batch_size": base_batch * base_grad_accum,
        "epochs": base_epochs,
        "learning_rate": base_lr,
        "lora_r": base_r,
        "lora_alpha": base_alpha,
        "context_window": context_window,
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "pad_to_multiple_of": pad_to_multiple_of,
        "target_updates": optimizer_steps_per_epoch(sample_count, base_batch, base_grad_accum) * base_epochs,
        "target_examples": sample_count * base_epochs,
    }

    if not auto_enabled:
        return profile

    reference_examples = max(1, int(auto_cfg.get("reference_examples", 48)))
    configured_min_epochs = max(1, int(auto_cfg.get("min_epochs", 4)))
    min_epochs = compute_adaptive_min_epochs(sample_count, configured_min_epochs)
    max_epochs = max(min_epochs, int(auto_cfg.get("max_epochs", max(base_epochs * 2, 60))))

    batch_profile = compute_auto_batch_profile(sample_count, context_window, runtime or {"name": runtime_name}, auto_cfg)
    batch_size = batch_profile["batch_size"]
    grad_accum = batch_profile["grad_accum"]
    batch_size, grad_accum = enforce_min_steps_per_epoch(sample_count, batch_size, grad_accum)
    batch_profile["batch_size"] = batch_size
    batch_profile["grad_accum"] = grad_accum

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
    target_examples = max(sample_count, target_updates * base_batch)
    epochs = math.ceil(target_examples / max(1, sample_count))
    epochs = max(min_epochs, min(max_epochs, epochs))

    profile.update(
        {
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "epochs": epochs,
            "learning_rate": min(base_lr, lr_cap),
            "lora_r": lora_r,
            "lora_alpha": max(base_alpha, lora_r * 2),
            "target_updates": target_updates,
            "target_examples": target_examples,
            "batch_profile": batch_profile,
        }
    )
    return profile


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    checkpoint_dir = resolve_project_path(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_names = [
        name
        for name in os.listdir(checkpoint_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, name))
    ]
    if not checkpoint_names:
        return None

    latest_name = max(checkpoint_names, key=lambda value: int(value.split("-")[1]))
    return os.path.join(checkpoint_dir, latest_name)


def compute_save_steps(total_train_steps: int, checkpoint_cfg: dict[str, Any]) -> int:
    requested = int(checkpoint_cfg.get("save_steps", 500))
    min_save_steps = max(1, int(checkpoint_cfg.get("min_save_steps", 1)))
    short_run_target_saves = max(1, int(checkpoint_cfg.get("short_run_target_saves", 2)))

    if total_train_steps <= 0:
        return max(min_save_steps, requested)

    if total_train_steps < requested:
        return max(min_save_steps, math.ceil(total_train_steps / short_run_target_saves))

    return max(min_save_steps, requested)


def compute_logging_steps(steps_per_epoch: int) -> int:
    return max(1, min(5, steps_per_epoch))
