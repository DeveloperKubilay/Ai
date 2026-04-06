import json
from typing import Any

import requests


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


def teacher_is_available(url: str, timeout: float = 3.0) -> bool:
    try:
        response = requests.get(url.rsplit("/", 1)[0] + "/tags", timeout=timeout)
        return response.ok
    except Exception:
        return False


def call_teacher_json_array(
    url: str,
    model_name: str,
    prompt: str,
    label: str,
    temperature: float,
    max_attempts: int,
    num_predict: int = 4000,
) -> list[Any]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    for attempt in range(max_attempts):
        try:
            response = requests.post(url, json=payload, timeout=240)
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
