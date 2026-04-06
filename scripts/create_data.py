import json
from hashlib import sha1
from typing import Any

import requests

from pipeline_utils import (
    build_single_turn_messages,
    get_system_prompt,
    load_config,
    normalize_messages,
    project_path,
    stable_json_dumps,
)


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


def extract_json_array(raw_text: str) -> list[Any]:
    cleaned = str(raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]") + 1
    if start == -1 or end <= start:
        raise ValueError("JSON array bulunamadi")

    payload = cleaned[start:end].replace("\r", " ").strip()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("JSON array bekleniyordu")
    return data


def call_teacher_json_array(
    prompt: str,
    label: str,
    model_name: str,
    temperature: float,
    max_attempts: int,
) -> list[Any]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 4000,
        },
    }

    for attempt in range(max_attempts):
        try:
            response = requests.post(teacher_cfg["url"], json=payload, timeout=240)
            response.raise_for_status()
            result = response.json().get("response", "")
            data = extract_json_array(result)
            if not data:
                raise ValueError("Bos JSON array dondu")
            return data
        except Exception as exc:
            print(f"  {label} hatasi ({attempt + 1}/{max_attempts}): {exc}")
            if attempt == max_attempts - 1:
                raise
            print(f"  {label} tekrar deneniyor...")

    raise ValueError(f"{label} cevap veremedi")


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

            try:
                if "messages" in item:
                    examples = build_direct_messages(item)
                    verified = True
                    ai_used = False
                else:
                    if item.get("ai") is False:
                        print("  ai:false ama hazir messages yok; bu kayit atlandi")
                        continue

                    content = fetch_content(item)
                    if not content:
                        print("  Kullanilabilir icerik bulunamadi, atlaniyor")
                        continue

                    examples = generate_examples_from_content(content)
                    examples, verified = verify_examples(content, examples)
                    ai_used = True
            except Exception as exc:
                print(f"  Isleme hatasi: {exc}")
                continue

            written_now = 0
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
print("python scripts/prepare_dataset.py")
print("python scripts/index.py")
