import os
from ultralytics import YOLO
import shutil
from datetime import datetime

# Загружаем модель
model = YOLO("yolov8n.pt")  # Используем предобученную модель YOLOv8n

# Пути к папкам
dataset_path = "/Users/vladpalamarchuk/Documents/Projects/dataset"
labeled_dir = "/Users/vladpalamarchuk/Documents/Projects/labeled_data"
os.makedirs(labeled_dir, exist_ok=True)  # Создаем папку для результатов

# Создаем подпапки для разных типов файлов
images_dir = os.path.join(labeled_dir, "images")
videos_dir = os.path.join(labeled_dir, "videos")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

# Получаем текущую дату и время для создания уникальных имен файлов
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Начинаем обработку файлов из {dataset_path}...")

# Счетчики для статистики
processed_images = 0
processed_videos = 0

# Обрабатываем все файлы в папке
for filename in os.listdir(dataset_path):
    # Пропускаем системные файлы и JSON
    if filename.startswith('.') or filename.endswith('.json'):
        continue
    
    file_path = os.path.join(dataset_path, filename)
    
    # Проверяем, что это файл, а не папка
    if not os.path.isfile(file_path):
        continue
    
    print(f"Обрабатываем файл: {filename}")
    
    # Определяем тип файла
    is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    is_video = filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    
    if is_image:
        # Выполняем детекцию на изображении
        results = model(file_path, conf=0.25)  # Устанавливаем порог уверенности 0.25
        
        # Создаем имя для выходного файла
        output_filename = f"labeled_{timestamp}_{filename}"
        output_path = os.path.join(images_dir, output_filename)
        
        # Сохраняем результат с разметкой
        results[0].save(filename=output_path)
        
        # Сохраняем информацию о детекции в JSON
        json_path = os.path.join(images_dir, f"{os.path.splitext(output_filename)[0]}.json")
        with open(json_path, 'w') as f:
            f.write(str(results[0].boxes.data.tolist()))
        
        processed_images += 1
        print(f"  Изображение обработано и сохранено в {output_path}")
        
    elif is_video:
        # Для видео создаем отдельную папку с именем видео
        video_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(videos_dir, f"{video_name}_{timestamp}")
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Выполняем детекцию на видео
        results = model.track(file_path, conf=0.25, save=True, project=video_output_dir, name="")
        
        processed_videos += 1
        print(f"  Видео обработано и сохранено в {video_output_dir}")
    
    else:
        print(f"  Пропускаем файл {filename} - неподдерживаемый формат")

print("\nОбработка завершена!")
print(f"Обработано изображений: {processed_images}")
print(f"Обработано видео: {processed_videos}")
print(f"\nРазмеченные изображения находятся в: {images_dir}")
print(f"Размеченные видео находятся в: {videos_dir}")