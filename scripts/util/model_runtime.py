import os
import warnings
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="google.protobuf.runtime_version")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.pipeline_utils import (
    call_with_dtype_fallback,
    detect_runtime,
    render_messages,
    resolve_model_reference,
    resolve_project_path,
)


def load_runtime_model(config: dict[str, Any], allow_base_fallback: bool = False) -> tuple[Any, Any]:
    model_path = resolve_project_path(config["model"]["output_dir"])
    base_model_id = resolve_model_reference(config["model"]["base_model"])
    runtime = detect_runtime()
    is_cuda = runtime["is_cuda"]

    tokenizer_path = model_path if os.path.exists(model_path) else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": runtime["dtype"],
    }
    if is_cuda:
        model_kwargs["device_map"] = "auto"

    base_model = call_with_dtype_fallback(
        AutoModelForCausalLM.from_pretrained,
        base_model_id,
        **model_kwargs,
    )

    if os.path.exists(model_path):
        model = PeftModel.from_pretrained(base_model, model_path)
    elif allow_base_fallback:
        model = base_model
    else:
        raise FileNotFoundError(f"Fine-tuned model bulunamadi: {model_path}")

    model.eval()
    return model, tokenizer


def generate_answer(
    config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int | None = None,
    do_sample: bool | None = None,
    repetition_penalty: float | None = None,
) -> str:
    inference_cfg = config.get("inference", {})
    prompt = render_messages(
        messages,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens or int(inference_cfg.get("max_new_tokens", 96)),
        do_sample=bool(inference_cfg.get("do_sample", False) if do_sample is None else do_sample),
        repetition_penalty=float(inference_cfg.get("repetition_penalty", 1.1) if repetition_penalty is None else repetition_penalty),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return answer.replace("<|im_end|>", "").replace("</s>", "").strip()
