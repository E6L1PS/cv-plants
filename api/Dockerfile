FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY .. /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]