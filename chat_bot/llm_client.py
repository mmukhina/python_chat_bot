# HTTP клиент для связи основного приложения с LLM сервисом.
import os
import requests

# Адрес, предел ожидания и глубина истории задаются окружением.
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5001")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "35"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))


# Своя ошибка, чтобы основной сервис мог обработать сбой LLM в одном месте.
class LLMError(Exception):
    pass


# Отправляет сообщения в LLM-сервис и возвращает ответ.
# Параметры: messages — список сообщений с полями role (автор) и content (текст).
def request_llm_reply(messages):
    try:
        # Передаём список сообщений в JSON. timeout ограничивает время ожидания.
        resp = requests.post(
            f"{LLM_SERVICE_URL}/api/generate",
            json={"messages": messages},
            timeout=LLM_TIMEOUT,
        )
    # Сервис слишком долго не отвечает.
    except requests.Timeout:
        raise LLMError("LLM service timeout")
    # Например, сервис выключен или нет соединения.
    except requests.RequestException as e:
        raise LLMError(f"LLM service unavailable: {e}")

    # Сохраняем причину ошибки вместо общего сообщения о статусе 502.
    try:
        # Пробуем прочитать ответ как JSON.
        data = resp.json()
    except ValueError:
        raise LLMError("LLM сервис вернул некорректный JSON")
    # При ошибочном статусе берём причину из ответа LLM сервиса.
    if not resp.ok:
        detail = "Ошибка LLM сервиса"
        if isinstance(data, dict):
            detail = data.get("error", detail)
        # Отдельно отмечаем превышение времени ожидания.
        if resp.status_code == 504:
            raise LLMError("LLM service timeout: " + str(detail))
        raise LLMError(str(detail))
    # Даже при успешном статусе проверяем, что модель прислала непустой текст.
    if not isinstance(data, dict) or not isinstance(data.get("content"), str) or not data["content"].strip():
        raise LLMError("Модель вернула пустой ответ. Повторите запрос.")
    return data
