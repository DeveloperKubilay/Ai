from util.model_runtime import generate_answer, load_runtime_model
from util.pipeline_utils import build_single_turn_messages, get_system_prompt, load_config


config = load_config()
system_prompt = get_system_prompt(config)

print("Model yukleniyor...")
model, tokenizer = load_runtime_model(config)
print("Model yuklendi!\n")

test_questions = [
    "Elenora nedir?",
    "Elenora nasıl kurulur?",
    "Elenora neden kullanılır?",
    "maxSize hangi birimdedir?",
    "newLog ne döndürür?",
    "Kurulum bilgisini JSON olarak ver.",
    "Atatürk Ferrari'ye bindi mi?",
]

for question in test_questions:
    print(f"\n{'=' * 60}")
    print(f"Soru: {question}")
    print(f"{'=' * 60}")

    answer = generate_answer(
        config,
        model,
        tokenizer,
        build_single_turn_messages(question, answer=None, system_prompt=system_prompt),
    )
    print(f"Cevap: {answer}")

print(f"\n{'=' * 60}")
print("Test tamamlandi!")
