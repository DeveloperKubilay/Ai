import os
import shutil

from pipeline_utils import (
    build_training_profile,
    compute_logging_steps,
    compute_save_steps,
    find_latest_checkpoint,
    get_checkpoint_dir,
    get_prepared_paths,
    load_config,
    optimizer_steps_per_epoch,
    read_json,
    write_json,
)

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


config = load_config()
training_cfg = config["training"]
preprocessing_cfg = config.get("preprocessing", {})
checkpoint_cfg = config.get("checkpointing", {})
seed = int(training_cfg.get("seed", 42))
set_seed(seed)

prepared_paths = get_prepared_paths(config, train_path="train.jsonl")
manifest = read_json(prepared_paths["manifest_path"])
if manifest is None or not manifest.get("complete") or not os.path.exists(prepared_paths["dataset_dir"]):
    raise SystemExit("Hazir token dataset bulunamadi. Once: python prepare_dataset.py")

print("=" * 60)
print("FINE-TUNING BASLIYOR")
print("=" * 60)
print(f"Run key: {prepared_paths['run_key']}")
print(f"Hazir dataset: {prepared_paths['dataset_dir']}")

dataset = load_from_disk(prepared_paths["dataset_dir"])
sample_count = len(dataset)
if sample_count == 0:
    raise ValueError("Hazir dataset bos.")

if "length" in dataset.column_names:
    token_lengths = list(dataset["length"])
else:
    token_lengths = [len(input_ids) for input_ids in dataset["input_ids"]]

profile = build_training_profile(sample_count, token_lengths, training_cfg, preprocessing_cfg)
avg_tokens = sum(token_lengths) / sample_count
steps_per_epoch = optimizer_steps_per_epoch(sample_count, profile["batch_size"], profile["grad_accum"])
save_steps = compute_save_steps(steps_per_epoch, checkpoint_cfg)
logging_steps = compute_logging_steps(steps_per_epoch)

tokenizer_source = config["model"].get("tokenizer_source", config["model"]["base_model"])
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
    f"epochs={profile['epochs']} | "
    f"lr={profile['learning_rate']} | "
    f"lora_r={profile['lora_r']}"
)
print(
    "Veri: "
    f"ornek={sample_count} | "
    f"ortalama_token={avg_tokens:.1f} | "
    f"max_token={max(token_lengths)} | "
    f"steps_per_epoch={steps_per_epoch}"
)

checkpoint_dir = get_checkpoint_dir(config, prepared_paths["run_key"])
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

model = AutoModelForCausalLM.from_pretrained(
    config["model"]["base_model"],
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
model.config.use_cache = False

if latest_checkpoint:
    print(f"Checkpointten devam ediliyor: {latest_checkpoint}")
    model = PeftModel.from_pretrained(model, latest_checkpoint)
else:
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

args = SFTConfig(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=profile["batch_size"],
    gradient_accumulation_steps=profile["grad_accum"],
    learning_rate=profile["learning_rate"],
    num_train_epochs=profile["epochs"],
    fp16=True,
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=int(checkpoint_cfg.get("save_total_limit", 2)),
    save_safetensors=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    report_to="none",
    seed=seed,
    group_by_length=True,
    max_length=profile["context_window"],
    shuffle_dataset=True,
    packing=False,
)

print(
    "Checkpoint: "
    f"dir={checkpoint_dir} | "
    f"save_steps={save_steps} | "
    f"save_total_limit={args.save_total_limit}"
)
print(f"Egitim basliyor (seed={seed})...\n")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    processing_class=tokenizer,
)

try:
    trainer.train(resume_from_checkpoint=latest_checkpoint)
except KeyboardInterrupt:
    print("\nEgitim durduruldu. Son checkpoint ile devam edebilirsin:")
    print("python index.py")
    raise

output_dir = config["model"]["output_dir"]
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
print("Test: python quick_test.py")
print("=" * 60)
