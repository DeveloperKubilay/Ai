import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================================
# YAPILANDIRMA
# ============================================

with open("settings.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# ============================================
# EĞİTİM
# ============================================

print("="*60)
print("🚀 FINE-TUNING BAŞLIYOR")
print("="*60)

# 1. Dataset yükle (stream)
print("📥 train.jsonl yükleniyor...")
dataset_list = []
with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        dataset_list.append(json.loads(line))

dataset = Dataset.from_list(dataset_list)
print(f"✅ {len(dataset)} örnek yüklendi")

# 2. Model yükle
print(f"📥 Model: {config['model']['base_model']}")
tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config["model"]["base_model"],
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 3. LoRA
lora_config = LoraConfig(
    r=config["training"]["lora_r"],
    lora_alpha=config["training"]["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("✅ LoRA uygulandı")

# 4. Eğitim
output_dir = config["model"]["output_dir"]

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config["training"]["batch_size"],
    gradient_accumulation_steps=4,
    learning_rate=config["training"]["learning_rate"],
    num_train_epochs=config["training"]["num_epochs"],
    fp16=True,
    logging_steps=5,
    save_strategy="no",  # Sadece sonda kaydet
    warmup_ratio=0.1,
    optim="adamw_torch",
    report_to="none"
)

print(f"🚀 Eğitim başlıyor ({config['training']['num_epochs']} epoch)...\n")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    formatting_func=lambda x: x["text"]
)

trainer.train()

# 5. Kaydet
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n" + "="*60)
print(f"✅ TAMAMLANDI! Model: {output_dir}")
print("🧪 Test: python quick_test.py")
print("="*60)
