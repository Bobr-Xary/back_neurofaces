import cv2
import glob
import os
import torch
from io import BytesIO
from gfpgan import GFPGANer
from basicsr.utils import imwrite
from PIL import Image


def upscale_with_gfpgan(image_path, model_path, upscale=2, version='1.3'):
    """Интерфейс для апскейлинга изображений с использованием GFPGAN"""
    # Установка устройства (CUDA или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка модели GFPGAN
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch='clean',  # Для версий '1.3' или '1.4'
        channel_multiplier=2,
        bg_upsampler=None  # Если фоновый апскейлинг не используется
    )

    # Загружаем изображение
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Применяем восстановление
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,  # Убедитесь, что изображения не выровнены
        only_center_face=False,
        paste_back=True
    )

    # Преобразуем восстановленное изображение в байты
    result_image = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB для корректного отображения
    pil_img = Image.fromarray(result_image)  # Преобразуем в формат PIL
    img_byte_arr = BytesIO()  # Используем BytesIO для преобразования в байты
    pil_img.save(img_byte_arr, format='PNG')  # Сохраняем в формате JPG
    img_byte_arr = img_byte_arr.getvalue()  # Получаем байтовое представление изображения

    return img_byte_arr

def process_images_in_folder(input_folder, output_folder, model_path, upscale=2):
    """Обрабатываем все изображения в папке"""
    img_list = sorted(glob.glob(os.path.join(input_folder, '*.jpg')))  # Обрабатываем только JPG файлы

    # Создание необходимых папок для сохранения результатов
    os.makedirs(os.path.join(output_folder, 'restored_imgs'), exist_ok=True)

    # Обработка всех изображений в папке
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f'Обрабатываем изображение: {img_name}')
        
        # Апскейл с помощью GFPGAN и перевод в байты
        img_bytes = upscale_with_gfpgan(img_path, model_path, upscale)

        # Сохранение результата в папке
        save_restore_path = os.path.join(output_folder, 'restored_imgs', img_name)
        with open(save_restore_path, 'wb') as f:
            f.write(img_bytes)  # Сохраняем изображение в байтах

    print(f'Обработка завершена. Результаты сохранены в {output_folder}')


if __name__ == '__main__':
    input_folder = 'reg_fol'  # Замените на путь к папке с изображениями
    output_folder = 'out'  # Замените на путь для сохранения результатов
    model_path = 'GFPGAN/gfpgan/weights/GFPGANv1.3.pth'  # Замените на путь к вашей модели GFPGAN

    process_images_in_folder(input_folder, output_folder, model_path, upscale=2)
