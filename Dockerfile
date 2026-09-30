FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN python src/train.py

# 8000: FastAPI inference API, 7860: Gradio UI
EXPOSE 8000 7860

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: run the API and the Gradio UI together in one container
# (docker run -p 8000:8000 -p 7860:7860 iris-mlops)
# CMD ["sh", "-c", "uvicorn src.serve:app --host 0.0.0.0 --port 8000 & python src/ui.py"]
