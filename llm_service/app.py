# FastAPI сервис генерации ответов Groq с тематической инструкцией.
import os
import logging
from typing import Annotated, Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException
from groq import Groq, APITimeoutError, APIConnectionError, APIError, AuthenticationError, RateLimitError

# Загружаем ключ и настройки из .env, если такой файл есть.
load_dotenv()
# Выводим сообщения о работе сервиса в терминал.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-service")
# Создаём отдельное приложение для обращения к модели.
app = FastAPI(title="Python LLM API", version="2.0.0", redoc_url=None)

# ---------- Конфигурация из env ----------

# Ключ читаем из окружения, в код его не записываем.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Выбираем модель, которая будет отвечать на вопросы.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
# Сколько секунд ждём ответ от Groq.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
# Максимальная длина нового сообщения в символах.
LLM_MAX_INPUT_LENGTH = int(os.getenv("LLM_MAX_INPUT_LENGTH", "4000"))
# Общий запас токенов на рассуждения модели и её ответ.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
# Настройка разнообразия ответов модели.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
# Эта инструкция задаёт тему бота: изучение Python.
LLM_TOPIC_INSTRUCTION = os.getenv(
    "LLM_TOPIC_INSTRUCTION",
    "Ты преподаватель Python. Помогай только с изучением Python, его библиотек и программированием на Python. На вопросы вне темы вежливо предложи вернуться к Python.",
)

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY не задан в .env — запросы к LLM будут падать с 502"
    )

# Создаём клиент Groq один раз. Без ключа вызов модели вернёт ошибку.
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT, max_retries=0)


# ---------- Вызов LLM ----------

# Отправляет сообщения модели Groq и получает текст ответа.
# Параметры: messages — список сообщений с полями role (автор) и content (текст).
def call_llm(messages):
    if client is None:
        raise RuntimeError("Не настроен GROQ_API_KEY")
    # GPT OSS расходует общий бюджет также на рассуждения.
    options = {}
    if LLM_MODEL.startswith("openai/gpt-oss"):
        options["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "low")
    try:
        # Отправляем системную инструкцию и историю переписки провайдеру.
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            extra_body=options,
        )
    # Ключ отсутствует, неверный или больше не действует.
    except AuthenticationError as e:
        logger.error("Groq auth error: %s", e)
        raise RuntimeError("Invalid API key")
    # Groq не успел ответить за заданное время.
    except APITimeoutError as e:
        logger.error("Groq timeout: %s", e)
        raise TimeoutError("LLM request timeout")
    # Превышен доступный лимит запросов к модели.
    except RateLimitError as e:
        logger.error("Groq rate limit: %s", e)
        raise RuntimeError("LLM rate limit exceeded, try again later")
    # Не удалось установить соединение. Возможна проблема с интернетом или VPN.
    except APIConnectionError as e:
        logger.error("Groq connection error: %s", e)
        raise RuntimeError("LLM connection failed")
    # Другие ошибки, которые вернул API Groq.
    except APIError as e:
        logger.error("Groq API error: %s", e)
        raise RuntimeError(f"LLM API error: {e}")

    # Из ответа Groq нам нужен только текст сообщения.
    content = ""
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""

    # Пустой ответ не считается успехом и не должен попадать в историю.
    if not content.strip():
        reason = "no_choices"
        if response.choices:
            reason = response.choices[0].finish_reason
        logger.warning("Empty model response, finish_reason=%s", reason)
        raise RuntimeError("Модель вернула пустой ответ. Попробуйте увеличить LLM_MAX_OUTPUT_TOKENS и повторить запрос.")

    # Основному сервису нужен только текст ответа.
    return {"content": content}


# Схемы Pydantic используются одновременно для проверки данных и OpenAPI.
# Описание одного сообщения в истории переписки.
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Annotated[str, StringConstraints(strict=True, min_length=1)]

# Основной сервис должен передать хотя бы одно сообщение.
class GenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

# Назад отправляем только текст ответа.
class GenerateOut(BaseModel):
    content: str

# Одинаковый формат ошибок для обоих сервисов.
class ErrorOut(BaseModel):
    error: str

# Обеспечивает совместимый JSON формат межсервисных ошибок.
# Параметры: request — HTTP запрос, который вызвал ошибку; exc — ошибка, которую нужно передать клиенту.
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

# Описывает неверные поля без раскрытия входного текста.
# Параметры: request — HTTP запрос, который вызвал ошибку; exc — ошибка, которую нужно передать клиенту.
@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = str(error["loc"][-1])
        errors.append(field + ": " + error["msg"])
    detail = "; ".join(errors)
    return JSONResponse({"error": detail}, status_code=422)

# Добавляет тематическую инструкцию и возвращает непустой ответ модели.
# Параметры: data — данные с историей переписки в поле messages.
@app.post("/api/generate", response_model=GenerateOut, tags=["Генерация"],
          responses={413:{"model":ErrorOut}, 422:{"model":ErrorOut}, 502:{"model":ErrorOut}, 504:{"model":ErrorOut}})
def generate(data: GenerateRequest):
    # Преобразуем объекты Pydantic в обычные словари для Groq.
    messages = []
    for message in data.messages:
        messages.append(message.model_dump())
    # Дополнительно отклоняем сообщения, состоящие только из пробелов.
    for message in messages:
        if not message["content"].strip():
            raise HTTPException(422, "Сообщение не должно быть пустым")
    # Проверяем длину последнего, то есть нового сообщения.
    if len(messages[-1]["content"]) > LLM_MAX_INPUT_LENGTH:
        raise HTTPException(413, "Превышена максимальная длина сообщения")
    # Ограничиваем размер всей истории, чтобы не отправлять слишком много текста.
    total_length = 0
    for message in messages:
        total_length += len(message["content"])
    if total_length > LLM_MAX_INPUT_LENGTH * 3:
        raise HTTPException(413, "Превышена максимальная длина контекста")
    # Просим отвечать кратко и использовать Markdown для оформления.
    style = os.getenv("LLM_RESPONSE_STYLE", "Отвечай кратко: обычно 2–5 предложений и небольшой пример кода при необходимости. Подробно объясняй только по просьбе пользователя. Используй Markdown и блоки кода с указанием языка.")
    # Системную инструкцию берём только с сервера, а не из присланной истории.
    full_messages = [{"role": "system", "content": LLM_TOPIC_INSTRUCTION + "\n" + style}]
    for message in messages:
        if message["role"] != "system":
            full_messages.append(message)
    try:
        return call_llm(full_messages)
    # Возвращаем основному сервису понятный статус тайм аута.
    except TimeoutError as exc:
        raise HTTPException(504, "LLM service timeout") from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    # Неожиданные ошибки записываем в лог, а клиенту показываем общее сообщение.
    except Exception as exc:
        logger.exception("Unexpected LLM error")
        raise HTTPException(502, "Ошибка обращения к модели") from exc

if __name__ == "__main__":
    import uvicorn
    # LLM сервис работает на порту 5001, отдельно от основного приложения.
    uvicorn.run(app, host="0.0.0.0", port=5001)
