# Проект по компьютерному зрению

Этот проект содержит инструменты для работы с моделями компьютерного зрения, включая Ruperta и YOLO.

## Структура проекта

```
/Data-Project/
├── dataset/            # Исходные данные для разметки (изображения, видео)
├── labeled_data/       # Результаты разметки
│   ├── images/         # Размеченные изображения
│   └── videos/         # Размеченные видео
├── venv/               # Виртуальное окружение
└── workspace/          # Рабочие скрипты и ноутбуки
    ├── Test-yolo.py    # Скрипт для работы с YOLO
    └── Test-ruberta.ipynb # Ноутбук для работы с Ruperta
```

## Настройка окружения

Виртуальное окружение настроено на уровне корневой папки проекта. Для активации:

```bash
source /Users/vladpalamarchuk/Documents/Data-Project/venv/bin/activate
```

## Основные зависимости

Основные пакеты, установленные в виртуальном окружении:

- ultralytics (8.3.86)
- torch (2.6.0)
- torchvision (0.21.0)
- transformers (4.49.0)
- numpy (1.26.4)
- opencv-python (4.11.0.86)
- matplotlib (3.10.1)
- scikit-learn (1.6.1)
- pandas (2.2.3)

## Использование скриптов

### Разметка данных с помощью YOLO

Скрипт `Test-yolo.py` используется для разметки изображений и видео с помощью модели YOLO:

```bash
cd /Users/vladpalamarchuk/Documents/Data-Project/workspace
python Test-yolo.py
```

Скрипт автоматически:
1. Обрабатывает все изображения и видео из папки `dataset`
2. Сохраняет размеченные изображения в `labeled_data/images/`
3. Сохраняет размеченные видео в `labeled_data/videos/`
4. Для изображений создает JSON-файлы с координатами обнаруженных объектов

### Работа с моделью Ruperta

Для работы с моделью Ruperta используйте ноутбук `Test-ruberta.ipynb`:

```bash
jupyter notebook /Users/vladpalamarchuk/Documents/Data-Project/workspace/Test-ruberta.ipynb
```

## Добавление новых данных

Для разметки новых данных:
1. Поместите файлы (изображения или видео) в папку `dataset`
2. Запустите скрипт `Test-yolo.py`
3. Результаты будут сохранены в папке `labeled_data` с уникальными именами

## Полезные команды Git

Для просмотра истории изменений:

```bash
# Общая история изменений
git log --stat --color=always | less -R

# История изменений конкретного файла
git log --follow --stat -- <filename>

# Просмотр различий между коммитами
git diff <commit_hash>..<commit_hash>
```
