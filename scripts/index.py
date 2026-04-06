import os
import shutil

from util.pipeline_utils import (
    build_checkpoint_resume_plan,
    build_eval_split_plan,
    build_training_profile,
    build_compatible_init_kwargs,
    call_with_dtype_fallback,
    compute_logging_steps,
    compute_save_steps,
    detect_runtime,
    find_latest_checkpoint,
    get_checkpoint_dir,
    get_prepared_paths,
    load_config,
    optimizer_steps_per_epoch,
    project_path,
    read_json,
    resolve_model_reference,
    resolve_project_path,
    write_json,
)

from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import SFTConfig, SFTTrainer


config = load_config()
training_cfg = config["training"]
preprocessing_cfg = config.get("preprocessing", {})
checkpoint_cfg = config.get("checkpointing", {})
evaluation_cfg = config.get("evaluation", {})
seed = int(training_cfg.get("seed", 42))
set_seed(seed)

prepared_paths = get_prepared_paths(config, train_path=project_path("data", "train.jsonl"))
manifest = read_json(prepared_paths["manifest_path"])
if manifest is None or not manifest.get("complete") or not os.path.exists(prepared_paths["dataset_dir"]):
    raise SystemExit("Hazir token dataset bulunamadi. Once: python scripts/prepare_dataset.py")

print("=" * 60)
print("FINE-TUNING BASLIYOR")
print("=" * 60)
print(f"Run key: {prepared_paths['run_key']}")
print(f"Hazir dataset: {prepared_paths['dataset_dir']}")

dataset = load_from_disk(prepared_paths["dataset_dir"])
raw_sample_count = len(dataset)
if raw_sample_count == 0:
    raise ValueError("Hazir dataset bos.")

checkpoint_dir = get_checkpoint_dir(config, prepared_paths["run_key"])
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

runtime = detect_runtime()
runtime_dtype = runtime["dtype"]
is_tpu = runtime["is_tpu"]
is_cuda = runtime["is_cuda"]

if is_tpu:
    print("TPU destegi algilandi! (bfloat16 kullanilacak)")
elif is_cuda:
    print(
        "CUDA/GPU modu algilandi! "
        f"({runtime.get('device_name', 'cuda')} | {runtime.get('total_memory_gb', 0):.1f} GB | float16)"
    )
else:
    print("CPU modu algilandi! (float32 kullanilacak)")

eval_plan = build_eval_split_plan(raw_sample_count, evaluation_cfg)
has_eval = eval_plan["enabled"]

if has_eval:
    split_dataset = dataset.train_test_split(test_size=eval_plan["eval_examples"], seed=seed, shuffle=True)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
else:
    train_dataset = dataset
    eval_dataset = None

train_sample_count = len(train_dataset)
if "length" in train_dataset.column_names:
    token_lengths = list(train_dataset["length"])
else:
    token_lengths = [len(input_ids) for input_ids in train_dataset["input_ids"]]

profile = build_training_profile(train_sample_count, token_lengths, training_cfg, preprocessing_cfg, runtime=runtime)
avg_tokens = profile["avg_tokens"]
steps_per_epoch = optimizer_steps_per_epoch(train_sample_count, profile["batch_size"], profile["grad_accum"])
total_train_steps = steps_per_epoch * profile["epochs"]
save_steps = compute_save_steps(total_train_steps, checkpoint_cfg)
logging_steps = compute_logging_steps(steps_per_epoch)
warmup_steps = max(1, int(total_train_steps * 0.1)) if total_train_steps > 0 else 0

tokenizer_source = resolve_model_reference(config["model"].get("tokenizer_source", config["model"]["base_model"]))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = profile["context_window"]

print(f"Model: {config['model']['base_model']}")
print(
    "Profil: "
    f"ctx={profile['context_window']} | "
    f"batch={profile['batch_size']} | "
    f"grad_accum={profile['grad_accum']} | "
    f"effective_batch={profile['effective_batch_size']} | "
    f"epochs={profile['epochs']} | "
    f"lr={profile['learning_rate']} | "
    f"lora_r={profile['lora_r']}"
)
print(
    "Veri: "
    f"egitim_ornek={train_sample_count} | "
    f"eval_ornek={len(eval_dataset) if eval_dataset is not None else 0} | "
    f"ortalama_token={avg_tokens:.1f} | "
    f"max_token={max(token_lengths)} | "
    f"steps_per_epoch={steps_per_epoch} | "
    f"toplam_step={total_train_steps}"
)

resume_plan = build_checkpoint_resume_plan(
    latest_checkpoint,
    profile,
    runtime,
    save_steps=save_steps,
    logging_steps=logging_steps,
)
resume_checkpoint = latest_checkpoint if resume_plan["resume_trainer_state"] else None

model_kwargs = {
    "trust_remote_code": True,
    "dtype": runtime_dtype,
}

if is_cuda:
    model_kwargs["device_map"] = "auto"

model = call_with_dtype_fallback(
    AutoModelForCausalLM.from_pretrained,
    resolve_model_reference(config["model"]["base_model"]),
    **model_kwargs,
)
model.config.use_cache = False

if latest_checkpoint:
    if resume_plan["load_adapter"]:
        model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=True)
        if resume_checkpoint:
            print(f"Checkpointten tam devam ediliyor: {latest_checkpoint}")
        else:
            print(f"Checkpoint bulundu, adapter yukleniyor: {latest_checkpoint}")
            if resume_plan["reasons"]:
                print("Optimizer/scheduler sifirdan kurulacak. Sebepler:")
                for reason in resume_plan["reasons"]:
                    print(f"- {reason}")
    else:
        print(f"Checkpoint bulundu ama mevcut profile uyumsuz, yok sayiliyor: {latest_checkpoint}")
        if resume_plan["reasons"]:
            print("Yeni LoRA egitimi baslatilacak. Sebepler:")
            for reason in resume_plan["reasons"]:
                print(f"- {reason}")
        latest_checkpoint = None
        resume_checkpoint = None

if not latest_checkpoint:
    lora_config = LoraConfig(
        r=profile["lora_r"],
        lora_alpha=profile["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("Yeni LoRA egitimi baslatiliyor")

sft_config_kwargs = build_compatible_init_kwargs(
    SFTConfig.__init__,
    {
        "output_dir": checkpoint_dir,
        "per_device_train_batch_size": profile["batch_size"],
        "gradient_accumulation_steps": profile["grad_accum"],
        "learning_rate": profile["learning_rate"],
        "num_train_epochs": profile["epochs"],
        "fp16": is_cuda,
        "bf16": is_tpu,
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": int(checkpoint_cfg.get("save_total_limit", 2)),
        "save_safetensors": True,
        "warmup_steps": warmup_steps,
        "optim": "adamw_torch",
        "report_to": "none",
        "seed": seed,
        "group_by_length": True,
        "max_length": profile["context_window"],
        "shuffle_dataset": True,
        "packing": False,
        "pad_to_multiple_of": profile["pad_to_multiple_of"],
        "gradient_checkpointing": not is_tpu,
        "dataloader_pin_memory": is_cuda,
    },
    label="SFTConfig",
)

if has_eval:
    sft_config_kwargs.update(
        build_compatible_init_kwargs(
            SFTConfig.__init__,
            {
                "eval_strategy": "steps",
                "eval_steps": save_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            },
            label="SFTConfig",
        )
    )

args = SFTConfig(
    **sft_config_kwargs
)

if is_tpu and hasattr(args, "gradient_checkpointing"):
    args.gradient_checkpointing = False

if is_tpu and hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

print(
    "Checkpoint: "
    f"dir={checkpoint_dir} | "
    f"save_steps={save_steps} | "
    f"save_total_limit={args.save_total_limit}"
)
if has_eval:
    print(
        "Eval: "
        f"every={save_steps} step | "
        f"patience={eval_plan['patience']} | "
        f"threshold={eval_plan['threshold']}"
    )
print(f"Egitim basliyor (seed={seed})...\n")

callbacks = []
if has_eval:
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=eval_plan["patience"],
            early_stopping_threshold=eval_plan["threshold"],
        )
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    processing_class=tokenizer,
    callbacks=callbacks,
)


def run_training(resume_from_checkpoint: str | None):
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)


try:
    run_training(resume_checkpoint)
except ValueError as exc:
    if resume_checkpoint and "parameter group" in str(exc):
        print("\nCheckpoint optimizer state bu profil ile uyusmadi. Adapter agirliklariyla sifirdan optimizer kuruluyor.")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        run_training(None)
    else:
        raise
except KeyboardInterrupt:
    print("\nEgitim durduruldu. Son checkpoint ile devam edebilirsin:")
    print("python scripts/index.py")
    raise

output_dir = resolve_project_path(config["model"]["output_dir"])
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

write_json(
    os.path.join(output_dir, "run_metadata.json"),
    {
        "prepared_run_key": prepared_paths["run_key"],
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "dataset_dir": os.path.abspath(prepared_paths["dataset_dir"]),
        "profile": profile,
        "source_manifest": manifest,
    },
)

if checkpoint_cfg.get("cleanup_after_success", False) and os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

print("\n" + "=" * 60)
print(f"TAMAMLANDI! Model: {output_dir}")
print("Test: python scripts/quick_test.py")
print("=" * 60)
