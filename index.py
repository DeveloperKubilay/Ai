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
n, max_len = len(dataset), max(len(x["text"]) for x in dataset)

# Auto config
ctx = next(x for x in [512, 1024, 2048, 4096, 8192] if x >= max_len)
epochs = 10 if n < 100 else 5 if n < 1000 else 3 if n < 10000 else 1
r = 8 if n < 100 else 16 if n < 10000 else 32
batch, grad_accum = 2, 8

print(f"📊 {n} örnek | ctx:{ctx} | epochs:{epochs} | r:{r} | lr:2e-4\n")

# Checkpoint kontrolü
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")] if os.path.exists(checkpoint_dir) else []
resume = os.path.join(checkpoint_dir, max(checkpoints, key=lambda x: int(x.split("-")[1]))) if checkpoints else None
if resume: print(f"🔄 Devam: {resume}")

# Model
tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["base_model"], 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

# LoRA
if resume:
    model = PeftModel.from_pretrained(model, resume)
else:
    lora_config = LoraConfig(
        r=r, 
        lora_alpha=r*2, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

# Eğitim
args = TrainingArguments(
    output_dir=checkpoint_dir, 
    per_device_train_batch_size=batch, 
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4, 
    num_train_epochs=epochs, 
    fp16=True, 
    logging_steps=5, 
    save_strategy="steps", 
    save_steps=50, 
    save_total_limit=2, 
    warmup_ratio=0.1, 
    optim="adamw_torch", 
    report_to="none"
)

trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset, 
    args=args, 
    max_seq_length=ctx, 
    formatting_func=lambda x: x["text"]
)

try:
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if os.path.exists(checkpoint_dir): shutil.rmtree(checkpoint_dir)
    print(f"✅ Tamamlandı: {output_dir}")
except KeyboardInterrupt:
    print(f"\n⚠️ Durduruldu! Devam: python index.py")
