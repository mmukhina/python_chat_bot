# Отдельное окружение сервиса обращения к модели Groq.
FROM python:3.12-slim
WORKDIR /app/llm_service
COPY llm_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY llm_service/ .
EXPOSE 5001
CMD ["python", "app.py"]
