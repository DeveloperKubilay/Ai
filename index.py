import torch, json, os, shutil
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

config = json.load(open("settings.json", "r", encoding="utf-8"))
checkpoint_dir, output_dir = "./checkpoints", config["model"]["output_dir"]

print("🚀 Eğitim başlıyor...")

# Dataset
dataset = Dataset.from_list([json.loads(line) for line in open("train.jsonl", "r", encoding="utf-8")])
print(f"✅ {len(dataset)} örnek")

# Checkpoint kontrolü
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")] if os.path.exists(checkpoint_dir) else []
resume = os.path.join(checkpoint_dir, max(checkpoints, key=lambda x: int(x.split("-")[1]))) if checkpoints else None
if resume: print(f"� Devam: {resume}")

# Model
tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config["model"]["base_model"], device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# LoRA
if resume:
    model = PeftModel.from_pretrained(model, resume)
else:
    model = get_peft_model(model, LoraConfig(r=config["training"]["lora_r"], lora_alpha=config["training"]["lora_alpha"], 
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

# Eğitim
args = TrainingArguments(output_dir=checkpoint_dir, per_device_train_batch_size=config["training"]["batch_size"], 
                         gradient_accumulation_steps=4, learning_rate=config["training"]["learning_rate"], 
                         num_train_epochs=config["training"]["num_epochs"], fp16=True, logging_steps=5, 
                         save_strategy="steps", save_steps=50, save_total_limit=2, warmup_ratio=0.1, 
                         optim="adamw_torch", report_to="none")

trainer = SFTTrainer(model=model, train_dataset=dataset, args=args, formatting_func=lambda x: x["text"])

try:
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if os.path.exists(checkpoint_dir): shutil.rmtree(checkpoint_dir)
    print(f"✅ Tamamlandı: {output_dir}")
except KeyboardInterrupt:
    print(f"\n⚠️ Durduruldu! Devam: python index.py")
