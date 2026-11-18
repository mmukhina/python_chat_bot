# Структура хранения диалогов и сообщений.
from datetime import datetime
from database import Base
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship


# ISO-строка с явным UTC-суффиксом Z, либо None.
# Параметры: dt — дата и время из базы; может быть None.
def _iso_utc(dt):
    if dt is None:
        return None
    # Буква Z показывает браузеру, что время указано в UTC.
    return dt.isoformat() + "Z"


# Один диалог объединяет сообщения; ORM удаляет их вместе с диалогом.
class Conversation(Base):
    __tablename__ = "conversations"

    # Уникальный номер записи, который база назначает сама.
    id = Column(Integer, primary_key=True)
    # В названии диалога можно хранить до 255 символов.
    title = Column(String(255), nullable=False, default="New conversation")
    # Время создания записи подставляется автоматически.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Время последнего изменения диалога.
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Связываем диалог с его сообщениями. При удалении диалога удаляем и их.
    messages = relationship(
        "Message", backref="conversation",
        cascade="all, delete-orphan", lazy="dynamic"
    )

    # Преобразует запись ORM в объект для JSON ответа.
    # Параметры: self — текущая запись диалога или сообщения.
    def to_dict(self):
        # Собираем обычный словарь, который можно отправить в JSON ответе.
        return {
            "id": self.id,
            "title": self.title,
            "created_at": _iso_utc(self.created_at),
            "updated_at": _iso_utc(self.updated_at),
        }


# Каждое сообщение хранит роль автора, текст и время отправки.
class Message(Base):
    __tablename__ = "messages"

    # Уникальный номер записи, который база назначает сама.
    id = Column(Integer, primary_key=True)
    # Номер диалога, которому принадлежит сообщение.
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    # Роль позволяет отличить текст пользователя от ответа модели.
    role = Column(String(20), nullable=False)  # user | assistant | system
    # Сам текст сообщения храним без ограничения размера столбца.
    content = Column(Text, nullable=False)
    # Время создания записи подставляется автоматически.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Преобразует запись ORM в объект для JSON ответа.
    # Параметры: self — текущая запись диалога или сообщения.
    def to_dict(self):
        # Собираем обычный словарь, который можно отправить в JSON ответе.
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": _iso_utc(self.created_at),
        }
