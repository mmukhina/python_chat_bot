# Python окружение основного сервиса и встроенного веб интерфейса.
FROM python:3.12-slim
WORKDIR /app/chat_bot
# Зависимости копируются отдельно для повторного использования слоя сборки.
COPY chat_bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# HTML, CSS и Python файлы находятся в одном каталоге.
COPY chat_bot/ .
EXPOSE 5002
CMD ["python", "app.py"]
