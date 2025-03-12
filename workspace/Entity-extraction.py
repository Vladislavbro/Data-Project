import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset


# Определяем директорию для сохранения размеченных данных
# Используем относительный путь от корня проекта
output_dir = os.path.join("..", "labeled_data", "named_entities")
# Проверяем, что директория существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"Результаты будут сохранены в: {os.path.abspath(output_dir)}")

# Шаг 2: Загружаем датасет conll2003
dataset = load_dataset("conll2003")
# Берём первые 5 примеров из обучающей части для простоты
sample_texts = dataset["train"]["tokens"][:5]  # Список токенов
sample_ids = dataset["train"]["id"][:5]        # ID для названий файлов

# Шаг 3: Загружаем предобученную модель для NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")

# Шаг 4: Обрабатываем тексты и извлекаем сущности
for idx, tokens in enumerate(sample_texts):
    # Преобразуем список токенов в строку
    text = " ".join(tokens)
    print(f"\nОбрабатываем текст {idx + 1}: {text}")
    
    # Извлекаем сущности
    entities = ner_pipeline(text)
    
    # Форматируем результаты
    result = f"Текст: {text}\nНайденные сущности:\n"
    for entity in entities:
        result += f"- {entity['word']} (Тип: {entity['entity']}, Score: {entity['score']:.4f})\n"
    
    # Выводим в консоль
    print(result)
    
    # Сохраняем в файл
    file_name = os.path.join(output_dir, f"result_{sample_ids[idx]}.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(result)

print(f"\nРезультаты сохранены в папку: {output_dir}")