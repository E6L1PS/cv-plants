# Курсовая работа по курсу Computer Vision

## Тема: "Приложение для сегментации изображения растений".

## Выполнил студент группы Пестерников Данил М80-214М-23

## Технологии

- **Uvicorn** — ASGI-сервер для запуска приложения.  
- **FastAPI** — фреймворк для создания API.  
- **PyTorch** — фреймворк для работы с нейронными сетями.  
- **PIL (Pillow)** — библиотека для работы с изображениями.  
- **OpenCV** — библиотека для обработки изображений.  
- **NumPy** — библиотека для работы с массивами данных. 

## Запуск и развёртывание

1. **Используя Docker:**  
    - Выполните команду:  
      ```bash
      docker compose up --build -d
      ```  
    - Обратите внимание: процесс сборки может занять продолжительное время, а итоговый образ может иметь размер более 10 ГБ.  

2. **Без Docker:**  
    - Перейдите в папку `api`:  
      ```bash
      cd ./api/
      ```  
    - Установите зависимости:  
      ```bash
      pip install -r requirements.txt
      ```  
    - Запустите сервер:  
      ```bash
      uvicorn main:app --reload
      ```  
Приложение будет доступно по `http://localhost:8000/`

### Примеры картинок тут:
```bash
cd ./plants_examples/
```  
## Скриншоты:
![sc1.png](plants_examples%2Fscreens%2Fsc1.png)
![sc2.png](plants_examples%2Fscreens%2Fsc2.png)
![sc3.png](plants_examples%2Fscreens%2Fsc3.png)
![sc4.png](plants_examples%2Fscreens%2Fsc4.png)

## Модель

В сервисе используется модель сегментации снимков растений. Модель была обучена на [датасете](https://www.kaggle.com/datasets/humansintheloop/plant-semantic-segmentation?resource=download)

В качестве архитектуры был взят MobileNet в схеме Encoder-Decoder. Модель обучалась на кросс-энтропии на задачу сегменации (или же многоклассовой классификации пикселей)

Оценивалась модель с помощью метрики mIoU (mean Intersection over Union). На валидационной выборке получили качество mIoU = 0.82.

Графики обучения модели можно найти в [cv_plants.ipynb](cv_plants.ipynb).