import os
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="google.protobuf.runtime_version")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_utils import load_config, resolve_model_reference, resolve_project_path


config = load_config()
model_path = resolve_project_path(config["model"]["output_dir"])
base_model_id = resolve_model_reference(config["model"]["base_model"])

print("Model yukleniyor...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

print("Model yuklendi!\n")

test_questions = [
    "Elenora nedir?",
    "Elenora nasıl kurulur?",
    "Elenora neden kullanılır?",
]

for question in test_questions:
    print(f"\n{'=' * 60}")
    print(f"Soru: {question}")
    print(f"{'=' * 60}")

    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=96,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    answer = answer.replace("<|im_end|>", "").strip()

    print(f"Cevap: {answer}")

print(f"\n{'=' * 60}")
print("Test tamamlandi!")
