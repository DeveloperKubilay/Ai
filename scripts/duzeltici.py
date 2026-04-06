import argparse
import json
import re
from difflib import SequenceMatcher
from hashlib import sha1
from pathlib import Path
from typing import Any

from util.model_runtime import generate_answer, load_runtime_model
from util.pipeline_utils import (
    build_single_turn_messages,
    get_system_prompt,
    load_config,
    normalize_messages,
    project_path,
    stable_json_dumps,
)
from util.teacher_client import call_teacher_json_array, teacher_is_available


WORD_RE = re.compile(r"[A-Za-z0-9_çğıöşüÇĞİÖŞÜ]+", re.UNICODE)
REFUSAL_MARKERS = (
    "verilen",
    "icerikte yer almiyor",
    "bilgi yok",
    "gecmiyor",
    "bilmiyorum",
    "emin degilim",
    "yer almiyor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Red-team ve repair set uretir.")
    parser.add_argument("--apply", action="store_true", help="Bulunan repair orneklerini train.jsonl'e ekle.")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


def build_train_qa_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs = []
    for record in records:
        try:
            messages = normalize_messages(
                record.get("messages", []),
                require_assistant=True,
                require_final_assistant=True,
            )
        except Exception:
            continue

        for index in range(len(messages) - 1):
            first = messages[index]
            second = messages[index + 1]
            if first["role"] == "user" and second["role"] == "assistant":
                pairs.append((first["content"], second["content"]))
    return pairs


def build_knowledge_context(pairs: list[tuple[str, str]]) -> str:
    seen = set()
    lines = []
    for question, answer in pairs:
        line = f"Soru: {question}\nCevap: {answer}"
        line_hash = sha1(line.encode("utf-8")).hexdigest()
        if line_hash in seen:
            continue
        seen.add(line_hash)
        lines.append(line)
    return "\n\n".join(lines[:40])


def build_domain_keywords(pairs: list[tuple[str, str]]) -> set[str]:
    keywords = {"elenora", "logger", "log", "rotation", "backupcount", "maxsize", "newlog", "console"}
    for question, answer in pairs:
        for token in tokenize_words(f"{question} {answer}"):
            if len(token) >= 4:
                keywords.add(token)
    return keywords


def fallback_questions() -> list[dict[str, str]]:
    return [
        {"question": "Atatürk Ferrari'ye bindi mi?", "kind": "scope"},
        {"question": "Elenora'nın React sürümü var mı?", "kind": "hallucination"},
        {"question": "Elenora hangi veritabanını kullanır?", "kind": "hallucination"},
        {"question": "backupCount tam olarak neyi belirler?", "kind": "paraphrase"},
        {"question": "maxSize karakter mi byte mı?", "kind": "paraphrase"},
        {"question": "newLog ne döndürür?", "kind": "paraphrase"},
        {"question": "Kurulum bilgisini JSON olarak ver.", "kind": "structured_output"},
        {"question": "Elenora Python paketi mi?", "kind": "scope"},
    ]


def build_redteam_questions(config: dict[str, Any], knowledge_context: str) -> list[dict[str, str]]:
    redteam_cfg = config.get("redteam", {})
    prompt_path = project_path("docs", "redteam_prompt.md")
    teacher_cfg = config.get("teacher", {})

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        prompt_template = ""

    base_questions = fallback_questions()

    if not prompt_template or not teacher_cfg.get("url") or not teacher_is_available(teacher_cfg["url"]):
        return base_questions

    try:
        prompt = prompt_template.format(
            knowledge_context=knowledge_context[:4000],
            question_count=max(4, int(redteam_cfg.get("question_count", 8))),
        )
        data = call_teacher_json_array(
            url=teacher_cfg["url"],
            model_name=redteam_cfg.get("model") or teacher_cfg["model"],
            prompt=prompt,
            label="Red-team",
            temperature=float(redteam_cfg.get("temperature", 0.5)),
            max_attempts=max(1, int(redteam_cfg.get("max_attempts", 2))),
        )
        questions = []
        for item in data:
            question = str((item or {}).get("question", "")).strip()
            kind = str((item or {}).get("kind", "paraphrase")).strip() or "paraphrase"
            if question:
                questions.append({"question": question, "kind": kind})
        merged = []
        seen = set()
        for item in [*questions, *base_questions]:
            key = item["question"].strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged
    except Exception as exc:
        print(f"Red-team teacher kullanilamadi, fallback'e donuluyor: {exc}")
        return base_questions


def best_matching_answer(question: str, pairs: list[tuple[str, str]]) -> tuple[str, float]:
    best_answer = ""
    best_score = 0.0
    for train_question, train_answer in pairs:
        score = SequenceMatcher(None, question.lower(), train_question.lower()).ratio()
        if score > best_score:
            best_score = score
            best_answer = train_answer
    return best_answer, best_score


def looks_like_refusal(answer: str) -> bool:
    lowered = str(answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(tokenize_words(left))
    right_tokens = set(tokenize_words(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(right_tokens))


def parse_json_like(text: str) -> Any | None:
    payload = str(text or "").strip()
    if not payload.startswith("{"):
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def heuristic_verdict(
    question: str,
    model_answer: str,
    kind: str,
    pairs: list[tuple[str, str]],
    domain_keywords: set[str],
    refusal_answer: str,
) -> dict[str, str]:
    expected_answer, question_score = best_matching_answer(question, pairs)
    question_tokens = set(tokenize_words(question))
    domain_overlap = question_tokens & domain_keywords

    if kind in {"scope", "hallucination"} or (not domain_overlap and "elenora" not in question.lower()):
        if looks_like_refusal(model_answer):
            return {"verdict": "pass", "category": "ok", "reason": "Kapsam disi soru dogru sekilde reddedildi.", "repaired_answer": ""}
        return {
            "verdict": "fail",
            "category": "scope",
            "reason": "Kapsam disi soruya uydurma veya fazla kesin cevap verildi.",
            "repaired_answer": refusal_answer,
        }

    if kind == "structured_output" and expected_answer.startswith("{") and not model_answer.strip().startswith("{"):
        return {
            "verdict": "fail",
            "category": "format",
            "reason": "Istenen JSON formatinda cevap donmedi.",
            "repaired_answer": expected_answer,
        }

    if kind == "structured_output" and expected_answer.startswith("{"):
        expected_json = parse_json_like(expected_answer)
        answer_json = parse_json_like(model_answer)
        if expected_json and answer_json:
            expected_keys = set(expected_json.keys())
            answer_keys = set(answer_json.keys())
            if not expected_keys or len(expected_keys & answer_keys) / max(1, len(expected_keys)) < 0.5:
                return {
                    "verdict": "fail",
                    "category": "format",
                    "reason": "JSON yapisi var ama beklenen alanlarla uyusmuyor.",
                    "repaired_answer": expected_answer,
                }

    if question_score >= 0.55 and expected_answer:
        answer_similarity = SequenceMatcher(None, model_answer.lower(), expected_answer.lower()).ratio()
        overlap = token_overlap_ratio(model_answer, expected_answer)
        if answer_similarity < 0.35 and overlap < 0.25:
            return {
                "verdict": "fail",
                "category": "wrong_fact",
                "reason": "Benzer egitim sorusuna gore cevap zayif veya yanlis.",
                "repaired_answer": expected_answer,
            }

    return {"verdict": "pass", "category": "ok", "reason": "Heuristik olarak kabul edildi.", "repaired_answer": ""}


def teacher_verify_cases(
    config: dict[str, Any],
    knowledge_context: str,
    cases: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    teacher_cfg = config.get("teacher", {})
    redteam_cfg = config.get("redteam", {})
    prompt_path = project_path("docs", "redteam_verify_prompt.md")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        return None

    if not prompt_template or not teacher_cfg.get("url") or not teacher_is_available(teacher_cfg["url"]):
        return None

    try:
        prompt = prompt_template.format(
            knowledge_context=knowledge_context[:4000],
            cases_json=json.dumps(cases, ensure_ascii=False, indent=2),
        )
        data = call_teacher_json_array(
            url=teacher_cfg["url"],
            model_name=redteam_cfg.get("verify_model") or config.get("verification", {}).get("model") or teacher_cfg["model"],
            prompt=prompt,
            label="Red-team verifier",
            temperature=float(redteam_cfg.get("verify_temperature", 0.2)),
            max_attempts=max(1, int(redteam_cfg.get("verify_attempts", 2))),
        )
        verdicts = []
        for item in data:
            verdicts.append(
                {
                    "verdict": str((item or {}).get("verdict", "fail")).strip().lower() or "fail",
                    "category": str((item or {}).get("category", "wrong_fact")).strip() or "wrong_fact",
                    "reason": str((item or {}).get("reason", "")).strip(),
                    "repaired_answer": str((item or {}).get("repaired_answer", "")).strip(),
                }
            )
        return verdicts if len(verdicts) == len(cases) else None
    except Exception as exc:
        print(f"Teacher verifier kullanilamadi, heuristik fallback devreye girdi: {exc}")
        return None


def merge_verdicts(
    teacher_info: dict[str, str] | None,
    heuristic_info: dict[str, str],
    kind: str,
    refusal_answer: str,
) -> dict[str, str]:
    if teacher_info is None:
        return heuristic_info

    merged = dict(teacher_info)
    merged.setdefault("verdict", "fail")
    merged.setdefault("category", "wrong_fact")
    merged.setdefault("reason", "")
    merged.setdefault("repaired_answer", "")

    if kind in {"scope", "hallucination"} and merged["verdict"] == "fail" and not looks_like_refusal(merged["repaired_answer"]):
        merged["category"] = "scope"
        merged["reason"] = merged["reason"] or "Kapsam disi soruda guvenli reddetme uygulanmali."
        merged["repaired_answer"] = refusal_answer

    if heuristic_info["verdict"] == "fail" and merged["verdict"] == "pass":
        return heuristic_info

    if merged["verdict"] == "fail" and not merged["repaired_answer"] and heuristic_info["repaired_answer"]:
        merged["repaired_answer"] = heuristic_info["repaired_answer"]

    return merged


def append_repairs(train_path: str, repair_rows: list[dict[str, Any]]) -> int:
    existing = read_jsonl(train_path)
    seen = {
        sha1(stable_json_dumps(row.get("messages", [])).encode("utf-8")).hexdigest()
        for row in existing
        if row.get("messages")
    }
    appended = 0
    with open(train_path, "a", encoding="utf-8") as f:
        for row in repair_rows:
            row_hash = sha1(stable_json_dumps(row["messages"]).encode("utf-8")).hexdigest()
            if row_hash in seen:
                continue
            seen.add(row_hash)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            appended += 1
    return appended


def main() -> None:
    args = parse_args()
    config = load_config()
    system_prompt = get_system_prompt(config)
    redteam_cfg = config.get("redteam", {})
    refusal_answer = redteam_cfg.get("refusal_answer", "Bu bilgi verilen Elenora içeriğinde yer almıyor.")
    train_path = project_path("data", "train.jsonl")
    report_path = project_path("data", "redteam_report.jsonl")
    repair_path = project_path("data", "repair_train.jsonl")

    train_records = read_jsonl(train_path)
    if not train_records:
        raise SystemExit("train.jsonl bos veya bulunamadi. Once: python scripts/create_data.py")

    qa_pairs = build_train_qa_pairs(train_records)
    if not qa_pairs:
        raise SystemExit("Red-team icin uygun soru-cevap cikarilamadi.")

    knowledge_context = build_knowledge_context(qa_pairs)
    domain_keywords = build_domain_keywords(qa_pairs)
    questions = build_redteam_questions(config, knowledge_context)

    print("=" * 60)
    print("RED-TEAM BASLIYOR")
    print("=" * 60)
    print(f"Soru sayisi: {len(questions)}")

    model, tokenizer = load_runtime_model(config, allow_base_fallback=False)

    cases = []
    for item in questions:
        question = item["question"]
        kind = item.get("kind", "paraphrase")
        model_answer = generate_answer(
            config,
            model,
            tokenizer,
            build_single_turn_messages(question, answer=None, system_prompt=system_prompt),
        )
        cases.append(
            {
                "question": question,
                "kind": kind,
                "model_answer": model_answer,
            }
        )

    teacher_verdicts = teacher_verify_cases(config, knowledge_context, cases)
    report_rows = []
    repair_rows = []

    for index, case in enumerate(cases):
        heuristic_info = heuristic_verdict(
            question=case["question"],
            model_answer=case["model_answer"],
            kind=case["kind"],
            pairs=qa_pairs,
            domain_keywords=domain_keywords,
            refusal_answer=refusal_answer,
        )
        teacher_info = teacher_verdicts[index] if teacher_verdicts else None
        verdict_info = merge_verdicts(teacher_info, heuristic_info, case["kind"], refusal_answer)

        report_row = {
            "question": case["question"],
            "kind": case["kind"],
            "model_answer": case["model_answer"],
            "verdict": verdict_info["verdict"],
            "category": verdict_info["category"],
            "reason": verdict_info["reason"],
            "repaired_answer": verdict_info["repaired_answer"],
        }
        report_rows.append(report_row)

        if verdict_info["verdict"] == "fail" and verdict_info["repaired_answer"]:
            repair_rows.append(
                {
                    "messages": normalize_messages(
                        [
                            {"role": "user", "content": case["question"]},
                            {"role": "assistant", "content": verdict_info["repaired_answer"]},
                        ],
                        system_prompt=system_prompt,
                        require_assistant=True,
                        require_final_assistant=True,
                    ),
                    "source": "redteam_repair",
                    "verified": True,
                    "ai_used": teacher_verdicts is not None,
                    "repair_kind": case["kind"],
                    "repair_category": verdict_info["category"],
                }
            )

    write_jsonl(report_path, report_rows)
    write_jsonl(repair_path, repair_rows)

    appended = 0
    if args.apply or redteam_cfg.get("apply_repairs", False):
        appended = append_repairs(train_path, repair_rows)

    fail_count = sum(1 for row in report_rows if row["verdict"] == "fail")
    print(f"Fail: {fail_count}/{len(report_rows)}")
    print(f"Repair ornegi: {len(repair_rows)}")
    if args.apply or redteam_cfg.get("apply_repairs", False):
        print(f"train.jsonl'e eklenen repair: {appended}")
    print(f"Rapor: {report_path}")
    print(f"Repair set: {repair_path}")


if __name__ == "__main__":
    main()
