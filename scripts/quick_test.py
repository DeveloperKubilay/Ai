import os
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="google.protobuf.runtime_version")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_utils import (
    build_single_turn_messages,
    get_system_prompt,
    load_config,
    render_messages,
    resolve_model_reference,
    resolve_project_path,
)


config = load_config()
system_prompt = get_system_prompt(config)
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

    prompt = render_messages(
        build_single_turn_messages(question, answer=None, system_prompt=system_prompt),
        tokenizer=tokenizer,
        add_generation_prompt=True,
        system_prompt=system_prompt,
    )
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
    answer = answer.replace("<|im_end|>", "").replace("</s>", "").strip()

    print(f"Cevap: {answer}")

print(f"\n{'=' * 60}")
print("Test tamamlandi!")
