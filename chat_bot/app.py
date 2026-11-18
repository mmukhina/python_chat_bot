# Основной FastAPI сервис: диалоги, история и веб интерфейс.
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException
from database import Base, engine, get_session
from models import Conversation, Message
from llm_client import request_llm_reply, LLMError, HISTORY_LIMIT

# Путь к папке с Python файлами и интерфейсом.
BASE_DIR = Path(__file__).resolve().parent
# Название должно быть строкой от 1 до 255 символов. Пробелы по краям убираются.
Title = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=255)]
# Пустое сообщение и текст длиннее 4000 символов не принимаются.
Content = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=4000)]

# Pydantic проверяет запросы и автоматически описывает их в Swagger.
# Данные для создания диалога. Если название не передали, используем стандартное.
class CreateConversation(BaseModel):
    title: Title = "Новый диалог"

# При переименовании новое название обязательно.
class RenameConversation(BaseModel):
    title: Title

# Из браузера получаем только текст нового сообщения.
class NewMessage(BaseModel):
    content: Content

# Какие данные диалога отправляем в браузер.
class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str | None

# Данные одного сообщения: автор, текст и время отправки.
class MessageOut(BaseModel):
    id: int
    conversation_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str

# При открытии диалога добавляем к его данным список сообщений.
class DialogueOut(ConversationOut):
    messages: list[MessageOut]

# После отправки возвращаем сразу сообщение пользователя и ответ бота.
class ReplyOut(BaseModel):
    conversation: ConversationOut
    user_message: MessageOut
    assistant_message: MessageOut

# Ошибки передаём в поле error, чтобы интерфейс мог их показать.
class ErrorOut(BaseModel):
    error: str

# Так сообщаем, что удаление прошло успешно.
class OkOut(BaseModel):
    ok: bool

# Создаёт отсутствующие таблицы, сохраняя существующую SQLite базу.
# Параметры: app — приложение FastAPI, которое запускается.
@asynccontextmanager
async def lifespan(app):
    # Создаём таблицы при первом запуске. Старые записи остаются на месте.
    Base.metadata.create_all(engine)
    # После этой строки приложение работает и принимает запросы.
    yield

# Создаём приложение. FastAPI сам добавляет Swagger по адресу /docs.
app = FastAPI(title="Python Chat Bot API", version="2.0.0", lifespan=lifespan,
              redoc_url=None, responses={404: {"model": ErrorOut}, 422: {"model": ErrorOut}})
# FastAPI будет давать каждому запросу своё подключение для работы с базой.

# Сохраняет формат error, который ожидает существующий JavaScript клиент.
# Параметры: request — HTTP запрос, который вызвал ошибку; exc — ошибка, которую нужно передать клиенту.
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

# Возвращает понятное описание неверного поля без содержимого сообщения.
# Параметры: request — HTTP запрос, который вызвал ошибку; exc — ошибка, которую нужно передать клиенту.
@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Собираем названия неверных полей и причины ошибок в одну строку.
    errors = []
    for error in exc.errors():
        field = str(error["loc"][-1])
        errors.append(field + ": " + error["msg"])
    detail = "; ".join(errors)
    return JSONResponse({"error": detail}, status_code=422)


# Находит диалог либо возвращает документированную ошибку 404.
# Параметры: session — сессия для чтения и сохранения данных в базе; conv_id — номер выбранного диалога в базе данных.
def find_conversation(session, conv_id):
    # Ищем запись в таблице диалогов по её id.
    conv = session.get(Conversation, conv_id)
    if conv is None:
        raise HTTPException(404, "Диалог не найден")
    return conv

# Возвращает диалоги по времени последнего изменения.
# Параметры: session — сессия для чтения и сохранения данных в базе.
@app.get("/api/conversations", response_model=list[ConversationOut], tags=["Диалоги"])
def list_conversations(session: Session = Depends(get_session)):
    # Новые и недавно изменённые диалоги показываем первыми.
    query = select(Conversation).order_by(Conversation.updated_at.desc())
    conversations = session.scalars(query).all()
    result = []
    for conversation in conversations:
        result.append(conversation.to_dict())
    return result

# Создаёт диалог с проверенным названием.
# Параметры: data — данные нового диалога с полем title (название); session — сессия для чтения и сохранения данных в базе.
@app.post("/api/conversations", response_model=ConversationOut, status_code=201, tags=["Диалоги"])
def create_conversation(data: CreateConversation, session: Session = Depends(get_session)):
    # Создаём объект диалога с названием из запроса.
    conv = Conversation(title=data.title)
    # Добавляем диалог в текущую сессию базы.
    session.add(conv)
    # Сохраняем изменения в базе данных.
    session.commit()
    return conv.to_dict()

# Возвращает диалог с сообщениями в хронологическом порядке.
# Параметры: conv_id — номер выбранного диалога в базе данных; session — сессия для чтения и сохранения данных в базе.
@app.get("/api/conversations/{conv_id}", response_model=DialogueOut, tags=["Диалоги"])
def get_conversation(conv_id: int, session: Session = Depends(get_session)):
    conv = find_conversation(session, conv_id)
    # Сначала идут старые сообщения. При одинаковом времени сравниваем id.
    messages = conv.messages.order_by(Message.created_at.asc(), Message.id.asc()).all()
    result = conv.to_dict()
    result["messages"] = []
    for message in messages:
        result["messages"].append(message.to_dict())
    return result

# Переименовывает выбранный диалог.
# Параметры: conv_id — номер выбранного диалога в базе данных; data — данные с новым названием диалога в поле title; session — сессия для чтения и сохранения данных в базе.
@app.patch("/api/conversations/{conv_id}", response_model=ConversationOut, tags=["Диалоги"])
def update_conversation(conv_id: int, data: RenameConversation, session: Session = Depends(get_session)):
    conv = find_conversation(session, conv_id)
    # Заменяем старое название на новое.
    conv.title = data.title
    # Сохраняем изменения в базе данных.
    session.commit()
    return conv.to_dict()

# Удаляет диалог вместе с его сообщениями.
# Параметры: conv_id — номер выбранного диалога в базе данных; session — сессия для чтения и сохранения данных в базе.
@app.delete("/api/conversations/{conv_id}", response_model=OkOut, tags=["Диалоги"])
def delete_conversation(conv_id: int, session: Session = Depends(get_session)):
    # Удаляем диалог. Его сообщения тоже удалятся благодаря настройке в models.py.
    session.delete(find_conversation(session, conv_id))
    # Сохраняем изменения в базе данных.
    session.commit()
    return {"ok": True}

# Отправляет историю модели и сохраняет сообщение вместе с ответом.
# Параметры: conv_id — номер выбранного диалога в базе данных; data — данные с текстом пользователя в поле content; session — сессия для чтения и сохранения данных в базе.
@app.post("/api/conversations/{conv_id}/messages", response_model=ReplyOut, tags=["Сообщения"],
          responses={502: {"model": ErrorOut}, 504: {"model": ErrorOut}})
def post_message(conv_id: int, data: NewMessage, session: Session = Depends(get_session)):
    conv = find_conversation(session, conv_id)
    # Готовим запись сообщения пользователя, связанного с выбранным диалогом.
    user = Message(conversation_id=conv.id, role="user", content=data.content)
    session.add(user)
    # Записываем сообщение в текущую транзакцию, но пока не подтверждаем сохранение.
    session.flush()
    # Выбираем последние сообщения средствами БД, исключая старые пустые ответы.
    history = session.scalars(select(Message).where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc(), Message.id.desc()).limit(max(1, HISTORY_LIMIT))).all()
    # Возвращаем историю в обычный порядок: от старых сообщений к новым.
    history.reverse()
    messages = []
    for message in history:
        if message.content.strip():
            messages.append({"role": message.role, "content": message.content})
    try:
        # Передаём историю отдельному LLM сервису и ждём ответ.
        result = request_llm_reply(messages)
    except LLMError as exc:
        # Если модель не ответила, отменяем добавление сообщения в базу.
        session.rollback()
        status = 502
        if "timeout" in str(exc).lower():
            status = 504
        raise HTTPException(status, str(exc)) from exc
    # Ответ модели сохраняем отдельным сообщением с ролью assistant.
    assistant = Message(conversation_id=conv.id, role="assistant", content=result["content"])
    session.add(assistant)
    # Если пользователь ещё не назвал диалог, берём начало первого сообщения.
    if conv.title in ("Новый диалог", "New conversation"):
        conv.title = data.content[:60]
        if len(data.content) > 60:
            conv.title += "..."
    # Обновляем время, чтобы этот диалог поднялся в списке.
    conv.updated_at = datetime.utcnow()
    # Сохраняем изменения в базе данных.
    session.commit()
    return {"conversation": conv.to_dict(), "user_message": user.to_dict(), "assistant_message": assistant.to_dict()}

# Возвращает HTML интерфейс приложения.
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")

# Отдаёт только публичные ресурсы, закрывая доступ к исходникам и базе.
# Параметры: filename — имя запрошенного файла интерфейса.
@app.get("/{filename}", include_in_schema=False)
def static_files(filename: str):
    # Разрешаем только перечисленные файлы, чтобы нельзя было скачать код или базу.
    if filename not in {"style.css"}:
        raise HTTPException(404, "Файл не найден")
    return FileResponse(BASE_DIR / filename)

if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер основного приложения на порту 5002.
    uvicorn.run(app, host="0.0.0.0", port=5002)
