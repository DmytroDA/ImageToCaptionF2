import os
from tkinter import Tk, Label, Button, Entry, filedialog, ttk
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Налаштування для використання моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gokaygokay/Florence-2-Flux", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("gokaygokay/Florence-2-Flux", trust_remote_code=True)

# Функція для генерації опису зображення
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Перевіряємо, чи зображення в RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        repetition_penalty=1.10,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

# Функція для вибору папки із зображеннями
def select_images_folder():
    folder_selected = filedialog.askdirectory(title="Виберіть папку із зображеннями")
    images_folder_entry.delete(0, 'end')
    images_folder_entry.insert(0, folder_selected)

# Функція для вибору папки для збереження підказок
def select_output_folder():
    folder_selected = filedialog.askdirectory(title="Виберіть папку для збереження підказок")
    output_folder_entry.delete(0, 'end')
    output_folder_entry.insert(0, folder_selected)

# Функція для генерації описів для всіх зображень
def generate_descriptions():
    images_folder = images_folder_entry.get()
    output_folder = output_folder_entry.get()

    if not images_folder or not output_folder:
        print("Будь ласка, вкажіть шляхи до папки із зображеннями та до папки для підказок.")
        return

    # Перевіряємо наявність папки із зображеннями
    if not os.path.exists(images_folder):
        print("Папка із зображеннями не знайдена!")
        return

    # Перевіряємо наявність папки для збереження підказок, створюємо, якщо не існує
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Отримуємо всі зображення у папці
    image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png", ".webp"))]

    # Налаштовуємо прогрес-бар
    progress_bar["maximum"] = len(image_files)
    progress_bar["value"] = 0

    # Проходимо по всіх зображеннях у папці
    for index, filename in enumerate(image_files):
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)

        # Генеруємо опис для кожного зображення
        description = run_example("<DESCRIPTION>", "Describe this image in great detail.", image)
        description_text = description["<DESCRIPTION>"]

        # Записуємо опис у файл з тим самим іменем, як у зображення
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(description_text)

        # Оновлюємо прогрес-бар
        progress_bar["value"] = index + 1
        root.update_idletasks()

        print(f"Підказка для {filename} збережена як {output_path}")

# Створюємо графічний інтерфейс
root = Tk()
root.title("Генератор підказок для зображень")

# Поле для введення шляху до папки із зображеннями
Label(root, text="Папка із зображеннями:").grid(row=0, column=0, padx=10, pady=10)
images_folder_entry = Entry(root, width=50)
images_folder_entry.grid(row=0, column=1, padx=10, pady=10)
Button(root, text="Вибрати...", command=select_images_folder).grid(row=0, column=2, padx=10, pady=10)

# Поле для введення шляху до папки для підказок
Label(root, text="Папка для підказок:").grid(row=1, column=0, padx=10, pady=10)
output_folder_entry = Entry(root, width=50)
output_folder_entry.grid(row=1, column=1, padx=10, pady=10)
Button(root, text="Вибрати...", command=select_output_folder).grid(row=1, column=2, padx=10, pady=10)

# Прогрес-бар
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=2, column=1, padx=10, pady=20)

# Кнопка для запуску генерації
Button(root, text="Генерувати підказки", command=generate_descriptions).grid(row=3, column=1, padx=10, pady=20)

# Запускаємо головний цикл програми
root.mainloop()
