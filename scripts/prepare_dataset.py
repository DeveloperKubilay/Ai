import hashlib
import json
import os
import shutil
from glob import glob

from pipeline_utils import (
    get_prepared_paths,
    load_config,
    project_path,
    read_json,
    resolve_model_reference,
    sha256_file,
    write_json,
)

from datasets import load_dataset
from transformers import AutoTokenizer


def load_existing_hashes(parts_dir: str) -> tuple[set[str], int]:
    seen_hashes = set()
    part_paths = sorted(glob(os.path.join(parts_dir, "part-*.jsonl")))

    for part_path in part_paths:
        with open(part_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text_hash = item.get("text_hash")
                if text_hash:
                    seen_hashes.add(text_hash)

    return seen_hashes, len(part_paths)


def create_manifest(config: dict, paths: dict[str, str], train_path: str, tokenizer_source: str) -> dict:
    return {
        "version": 2,
        "run_id": paths["run_id"],
        "run_key": paths["run_key"],
        "source_path": os.path.abspath(train_path),
        "source_sha256": sha256_file(train_path),
        "tokenizer_source": tokenizer_source,
        "next_line_index": 0,
        "part_count": 0,
        "unique_examples": 0,
        "empty_skipped": 0,
        "duplicate_skipped": 0,
        "invalid_skipped": 0,
        "token_length_sum": 0,
        "token_length_max": 0,
        "complete": False,
        "prepared_root": os.path.abspath(paths["root_dir"]),
        "dataset_dir": os.path.abspath(paths["dataset_dir"]),
        "base_model": config["model"]["base_model"],
        "loss_mode": "assistant_only",
    }


def split_prompt_and_answer(text: str) -> tuple[str, str]:
    assistant_marker = "<|im_start|>assistant\n"
    marker_index = text.rfind(assistant_marker)
    if marker_index == -1:
        raise ValueError("assistant bolumu bulunamadi")

    prompt_text = text[: marker_index + len(assistant_marker)]
    answer_text = text[marker_index + len(assistant_marker) :]
    if not answer_text.strip():
        raise ValueError("assistant cevabi bos")

    return prompt_text, answer_text


def flush_chunk(parts_dir: str, part_index: int, rows: list[dict]) -> str:
    os.makedirs(parts_dir, exist_ok=True)
    final_path = os.path.join(parts_dir, f"part-{part_index:06d}.jsonl")
    temp_path = final_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(temp_path, final_path)
    return final_path


def rebuild_arrow_dataset(parts_dir: str, dataset_dir: str) -> int:
    part_paths = sorted(glob(os.path.join(parts_dir, "part-*.jsonl")))
    if not part_paths:
        raise ValueError("Tokenize edilmis parca bulunamadi.")

    dataset = load_dataset("json", data_files=part_paths, split="train")
    if "text_hash" in dataset.column_names:
        dataset = dataset.remove_columns("text_hash")

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    dataset.save_to_disk(dataset_dir)
    return len(dataset)


config = load_config()
train_path = project_path("data", "train.jsonl")
preprocessing_cfg = config.get("preprocessing", {})
tokenizer_source = resolve_model_reference(config["model"].get("tokenizer_source", config["model"]["base_model"]))
chunk_size = max(1, int(preprocessing_cfg.get("chunk_size", 128)))
paths = get_prepared_paths(config, train_path=train_path)

os.makedirs(paths["run_dir"], exist_ok=True)
os.makedirs(paths["parts_dir"], exist_ok=True)

manifest = read_json(paths["manifest_path"])
if manifest is None:
    manifest = create_manifest(config, paths, train_path, tokenizer_source)

if manifest.get("complete") and os.path.exists(paths["dataset_dir"]):
    print(f"Hazir dataset zaten var: {paths['dataset_dir']}")
    print(f"Run key: {paths['run_key']}")
    raise SystemExit(0)

seen_hashes, existing_part_count = load_existing_hashes(paths["parts_dir"])
manifest["part_count"] = max(int(manifest.get("part_count", 0)), existing_part_count)
write_json(paths["manifest_path"], manifest)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
eos_token_id = tokenizer.eos_token_id

print("=" * 60)
print("TOKENIZATION BASLIYOR")
print("=" * 60)
print(f"Run key: {paths['run_key']}")
print(f"Kaynak: {os.path.abspath(train_path)}")
print(f"Tokenize edilecek benzersiz satirlar icin chunk_size={chunk_size}")
if manifest["next_line_index"] > 0 or existing_part_count > 0:
    print(
        "Devam modu: "
        f"line>{manifest['next_line_index']} | "
        f"part={manifest['part_count']} | "
        f"gorulen_ornek={len(seen_hashes)}"
    )

buffer = []
last_line_index = manifest["next_line_index"]
part_index = manifest["part_count"]
unique_examples = int(manifest.get("unique_examples", 0))
empty_skipped = int(manifest.get("empty_skipped", 0))
duplicate_skipped = int(manifest.get("duplicate_skipped", 0))
invalid_skipped = int(manifest.get("invalid_skipped", 0))
token_length_sum = int(manifest.get("token_length_sum", 0))
token_length_max = int(manifest.get("token_length_max", 0))

with open(train_path, "r", encoding="utf-8") as f:
    for line_index, line in enumerate(f, 1):
        if line_index <= manifest["next_line_index"]:
            continue

        last_line_index = line_index
        line = line.strip()
        if not line:
            empty_skipped += 1
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            invalid_skipped += 1
            continue

        text = item.get("text", "").strip()
        if not text:
            empty_skipped += 1
            continue

        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            duplicate_skipped += 1
            continue

        prompt_text, answer_text = split_prompt_and_answer(text)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids
        if eos_token_id is not None and (not input_ids or input_ids[-1] != eos_token_id):
            input_ids.append(eos_token_id)
            answer_ids.append(eos_token_id)

        labels = [-100] * len(prompt_ids) + answer_ids

        token_length = len(input_ids)
        token_length_sum += token_length
        token_length_max = max(token_length_max, token_length)
        unique_examples += 1
        seen_hashes.add(text_hash)
        buffer.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "length": token_length,
                "text_hash": text_hash,
            }
        )

        if len(buffer) >= chunk_size:
            flush_chunk(paths["parts_dir"], part_index, buffer)
            part_index += 1
            buffer = []
            manifest.update(
                {
                    "next_line_index": last_line_index,
                    "part_count": part_index,
                    "unique_examples": unique_examples,
                    "empty_skipped": empty_skipped,
                    "duplicate_skipped": duplicate_skipped,
                    "invalid_skipped": invalid_skipped,
                    "token_length_sum": token_length_sum,
                    "token_length_max": token_length_max,
                    "complete": False,
                }
            )
            write_json(paths["manifest_path"], manifest)
            print(
                "Kaydedildi: "
                f"line={last_line_index} | "
                f"part={part_index} | "
                f"unique={unique_examples}"
            )

if buffer:
    flush_chunk(paths["parts_dir"], part_index, buffer)
    part_index += 1

manifest.update(
    {
        "next_line_index": last_line_index,
        "part_count": part_index,
        "unique_examples": unique_examples,
        "empty_skipped": empty_skipped,
        "duplicate_skipped": duplicate_skipped,
        "invalid_skipped": invalid_skipped,
        "token_length_sum": token_length_sum,
        "token_length_max": token_length_max,
        "complete": False,
    }
)
write_json(paths["manifest_path"], manifest)

dataset_size = rebuild_arrow_dataset(paths["parts_dir"], paths["dataset_dir"])
manifest.update(
    {
        "dataset_size": dataset_size,
        "complete": True,
    }
)
write_json(paths["manifest_path"], manifest)

avg_length = token_length_sum / unique_examples if unique_examples else 0.0

print("\n" + "=" * 60)
print("TOKENIZATION TAMAMLANDI")
print("=" * 60)
print(f"Hazir dataset: {paths['dataset_dir']}")
print(f"Benzersiz ornek: {dataset_size}")
print(f"Duplicate atlandi: {duplicate_skipped}")
print(f"Bos atlandi: {empty_skipped}")
print(f"Gecersiz atlandi: {invalid_skipped}")
print(f"Ortalama token: {avg_length:.1f}")
print(f"Maks token: {token_length_max}")
