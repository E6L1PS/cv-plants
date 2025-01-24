import os
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pt', map_location=device)
model.to(device)
model.eval()

app = FastAPI()


def preprocess_image(image: Image.Image):
    """
    Предобработка изображения для подачи в модель.

    Преобразует изображение в тензор, изменяет размер и нормализует значения.

    :param image: Изображение в формате PIL.
    :return: Тензор изображения, готовый для инференса модели.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((768, 1152)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return transform(image).unsqueeze(0).to(device)


def postprocess_output(output, original_image):
    """
     Постобработка выходной маски, возвращаемой моделью, с учетом размера исходного изображения.

     Преобразует результат инференса модели в маску и масштабирует её до размера исходного изображения.

     :param output: Результат инференса модели (логиты).
     :param original_image: Оригинальное изображение в формате PIL.
     :return: Маска, масштабированная до размера исходного изображения.
     """

    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    scaled_mask = np.clip(output * 255 / 2, 0, 255).astype(np.uint8)
    original_width, original_height = original_image.size
    resized_mask = cv2.resize(scaled_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    return resized_mask


def postprocess_output_with_boxes(output, original_image):
    """
    Постобработка результата сегментации с добавлением боксов на оригинальное изображение.

    Получает сегментированную маску, находит контуры, и рисует ограничивающие прямоугольники на исходном изображении.

    :param output: Результат инференса модели (логиты).
    :param original_image: Оригинальное изображение в формате PIL.
    :return: Оригинальное изображение с нарисованными прямоугольниками вокруг сегментированных объектов.
    """

    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    segmented_mask = (output * 255 / 2).astype(np.uint8)
    original_width, original_height = original_image.size
    resized_mask = cv2.resize(segmented_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image_cv, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return original_image_cv


@app.get("/", response_class=HTMLResponse)
async def get_html():
    """
       Возвращает HTML страницу с интерфейсом для загрузки изображений.

       :return: HTML страница с возможностью загрузки изображения.
       """
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, headers={"Content-Type": "text/html; charset=utf-8"})


@app.post("/segment")
async def segment_image(file: UploadFile = File(...), mode: str = Form(...)):
    """
    Сегментация изображения, загруженного пользователем.

    В зависимости от режима ('boxes' или 'mask'), возвращает результат сегментации с боксами или маской.

    :param file: Загрузка изображения для сегментации.
    :param mode: Режим сегментации ('boxes' для прямоугольников, 'mask' для маски).
    :return: Путь к сохраненному результату сегментации.
    """

    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "Invalid file type. Only JPEG and PNG are allowed."}

        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)

        saved_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(saved_dir, exist_ok=True)

        if mode == "boxes":
            result_image = postprocess_output_with_boxes(output, image)
            save_path = os.path.join(saved_dir, f"segmented_with_boxes_{file.filename}")
            cv2.imwrite(save_path, result_image)
        elif mode == "mask":
            segmented_mask = postprocess_output(output, image)
            colored_mask = cv2.applyColorMap(segmented_mask, cv2.COLORMAP_HSV)
            result_image = Image.fromarray(colored_mask)
            save_path = os.path.join(saved_dir, f"segmented_mask_{file.filename}")
            result_image.save(save_path, format="PNG")
        else:
            return {"error": "Invalid mode. Use 'boxes' or 'mask'."}

        return {"saved_path": os.path.basename(save_path)}

    except FileNotFoundError as e:
        return {"error": f"File not found: {str(e)}"}
    except IOError as e:
        return {"error": f"Error processing image: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@app.get("/results/{file_path}")
async def get_result_image(file_path: str):
    """
      Возвращает результат сегментации по запросу пользователя.

      :param file_path: Путь к файлу с результатом сегментации.
      :return: Изображение с результатом сегментации.
      """
    return FileResponse(os.path.join("results", file_path), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
