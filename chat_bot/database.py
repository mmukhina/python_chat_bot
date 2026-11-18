# Соединение SQLAlchemy и отдельная сессия на каждый HTTP запрос.
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# В Docker адрес базы задаётся в compose.yaml. Локально используем файл chat.db.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{Path(__file__).with_name('chat.db')}")
# SQLite разрешаем использовать из разных потоков, в которых FastAPI выполняет запросы.
connection_settings = {}
if DATABASE_URL.startswith("sqlite"):
    connection_settings["check_same_thread"] = False
engine = create_engine(DATABASE_URL, connect_args=connection_settings)
# Создаём сессии для чтения и записи. Данные объектов доступны и после commit.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
# От этого класса наследуются таблицы Conversation и Message.
Base = declarative_base()


# Открывает сессию базы на время одного запроса.
def get_session():
    # Открываем сессию на время одного запроса.
    with SessionLocal() as session:
        # Передаём сессию маршруту. После его завершения with закроет её.
        yield session
