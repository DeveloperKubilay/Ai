import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "./qwen-trained-model"

print("🔄 Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

print("✅ Model yüklendi!\n")

# Test soruları
test_questions = [
    "Elenora nedir?",
    "Elenora nasıl yüklenir?",
    "Elenora ne işe yarar?"
]

for soru in test_questions:
    print(f"\n{'='*60}")
    print(f"❓ Soru: {soru}")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": soru}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        temperature=0.3,  # Daha deterministik
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.2,  # Tekrarları azalt
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    cevap = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"✨ Cevap: {cevap}")

print(f"\n{'='*60}")
print("✅ Test tamamlandı!")
