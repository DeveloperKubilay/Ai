import json
import os
from hashlib import sha1
from typing import Any

import requests

from util.pipeline_utils import (
    build_single_turn_messages,
    get_system_prompt,
    load_config,
    normalize_messages,
    project_path,
    stable_json_dumps,
)
from util.teacher_client import call_teacher_json_array


config = load_config()
system_prompt = get_system_prompt(config)
teacher_cfg = config.get("teacher", {})
verification_cfg = config.get("verification", {})
teacher_prompt_path = project_path("docs", "teacher_prompt.md")
teacher_verify_prompt_path = project_path("docs", "teacher_verify_prompt.md")
input_path = project_path("data", "input.jsonl")
train_path = project_path("data", "train.jsonl")

with open(teacher_prompt_path, "r", encoding="utf-8") as f:
    teacher_prompt = f.read()

try:
    with open(teacher_verify_prompt_path, "r", encoding="utf-8") as f:
        teacher_verify_prompt = f.read()
except FileNotFoundError:
    teacher_verify_prompt = ""

def normalize_example_record(raw_example: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw_example, dict):
        return None

    try:
        if "messages" in raw_example:
            messages = normalize_messages(
                raw_example["messages"],
                system_prompt=system_prompt,
                require_assistant=True,
                require_final_assistant=True,
            )
        else:
            question = raw_example.get("soru", raw_example.get("question", raw_example.get("user")))
            answer = raw_example.get("cevap", raw_example.get("answer", raw_example.get("assistant")))
            if not str(question or "").strip() or not str(answer or "").strip():
                return None
            messages = build_single_turn_messages(str(question).strip(), str(answer).strip(), system_prompt=system_prompt)
    except ValueError:
        return None

    return {"messages": messages}


def normalize_example_list(raw_examples: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    seen_hashes = set()

    for raw_example in raw_examples:
        record = normalize_example_record(raw_example)
        if record is None:
            continue

        message_hash = sha1(stable_json_dumps({"messages": record["messages"]}).encode("utf-8")).hexdigest()
        if message_hash in seen_hashes:
            continue
        seen_hashes.add(message_hash)
        normalized.append(record)

    return normalized


def fetch_content(item: dict[str, Any]) -> str | None:
    if "url" in item:
        print(f"  URL: {item['url']}")
        response = requests.get(item["url"], timeout=20)
        response.raise_for_status()
        try:
            data = response.json()
            if "readme" in data:
                return f"{data.get('description', '')}\n\n{data.get('readme', '')[:2000]}"
            return json.dumps(data, ensure_ascii=False)[:2000]
        except Exception:
            return response.text[:2000]

    if "text" in item:
        print(f"  Text: {len(item['text'])} karakter")
        return str(item["text"])[:2000]

    return None


def generate_examples_from_content(content: str) -> list[dict[str, Any]]:
    prompt = teacher_prompt.format(
        num_questions=config["training"]["num_questions"],
        content=content[:2000],
    )
    print("  Teacher'a gonderiliyor...")
    raw_examples = call_teacher_json_array(
        url=teacher_cfg["url"],
        prompt=prompt,
        label="Teacher",
        model_name=teacher_cfg["model"],
        temperature=float(teacher_cfg.get("temperature", 0.7)),
        max_attempts=3,
    )
    normalized_examples = normalize_example_list(raw_examples)
    print(f"  {len(normalized_examples)} temiz ornek uretildi")
    return normalized_examples


def verify_examples(content: str, examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if not examples:
        return [], False

    if not verification_cfg.get("enabled", False) or not teacher_verify_prompt:
        return examples, False

    verify_prompt = teacher_verify_prompt.format(
        content=content[:2000],
        examples_json=json.dumps(examples, ensure_ascii=False, indent=2),
    )
    print("  Verifier'a gonderiliyor...")

    try:
        raw_examples = call_teacher_json_array(
            url=teacher_cfg["url"],
            prompt=verify_prompt,
            label="Verifier",
            model_name=verification_cfg.get("model") or teacher_cfg["model"],
            temperature=float(verification_cfg.get("temperature", 0.2)),
            max_attempts=max(1, int(verification_cfg.get("max_attempts", 2))),
        )
        normalized_examples = normalize_example_list(raw_examples)
        if normalized_examples:
            print(f"  {len(normalized_examples)} ornek verifier'dan gecti")
            return normalized_examples, True
    except Exception as exc:
        print(f"  Verifier kullanilamadi: {exc}")

    if verification_cfg.get("strict", False):
        print("  Strict verification acik oldugu icin ornekler atlandi")
        return [], False

    print("  Verifier gecersiz dondu, orijinal teacher ornekleri kullaniliyor")
    return examples, False


def build_direct_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" not in item:
        return []

    messages = normalize_messages(
        item["messages"],
        system_prompt=system_prompt,
        require_assistant=True,
        require_final_assistant=True,
    )
    print("  Hazir messages verisi dogrudan kullaniliyor")
    return [{"messages": messages}]


def load_file_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    file_value = str(item.get("file", "")).strip()
    if not file_value:
        return []

    file_path = project_path(*file_value.replace("\\", "/").split("/")) if not os.path.isabs(file_value) else file_value
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            child = json.loads(line)
            if "ai" not in child and "ai" in item:
                child["ai"] = item["ai"]
            records.append(child)
    return records


print("=" * 60)
print("INPUT.JSONL ISLENIYOR")
print("=" * 60)

total_written = 0
dedupe_hashes = set()

with open(train_path, "w", encoding="utf-8") as train_file:
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            print(f"\n[{line_num}] Isleniyor...")
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print("  Gecersiz JSON satiri, atlaniyor")
                continue

            source_items = [item]
            if "file" in item:
                try:
                    source_items = load_file_records(item)
                    print(f"  Dosyadan {len(source_items)} alt kayit yuklendi")
                except Exception as exc:
                    print(f"  Dosya okunamadi: {exc}")
                    continue

            written_now = 0
            for source_item in source_items:
                try:
                    if "messages" in source_item:
                        examples = build_direct_messages(source_item)
                        verified = True
                        ai_used = False
                    else:
                        if source_item.get("ai") is False:
                            print("  ai:false ama hazir messages yok; bu kayit atlandi")
                            continue

                        content = fetch_content(source_item)
                        if not content:
                            print("  Kullanilabilir icerik bulunamadi, atlaniyor")
                            continue

                        examples = generate_examples_from_content(content)
                        examples, verified = verify_examples(content, examples)
                        ai_used = True
                except Exception as exc:
                    print(f"  Isleme hatasi: {exc}")
                    continue

                for example in examples:
                    record = {
                        "messages": example["messages"],
                        "source": "teacher" if ai_used else "input",
                        "verified": verified if ai_used else True,
                        "ai_used": ai_used,
                    }
                    record_hash = sha1(stable_json_dumps(record["messages"]).encode("utf-8")).hexdigest()
                    if record_hash in dedupe_hashes:
                        continue

                    dedupe_hashes.add(record_hash)
                    train_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written_now += 1

            total_written += written_now
            print(f"  {written_now} ornek data/train.jsonl'e eklendi")

print("\n" + "=" * 60)
print(f"TAMAMLANDI! Toplam {total_written} ornek")
print("=" * 60)
print("Siradaki adimlar:")
print("python scripts/veri_hazirla.py")
print("python scripts/egit.py")
