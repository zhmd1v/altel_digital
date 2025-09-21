# bot.py
#hello
import os
import io
import logging
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from get_response import process_video_comments

import requests
import pandas as pd
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command

# ---------- Конфиг и логирование ----------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
YT_API_KEY = os.getenv("YT_API_KEY", "")


if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN в .env")
if not YT_API_KEY:
    logging.warning("⚠️ Не задан YT_API_KEY в .env — парсинг YouTube работать не будет.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("yt-bot")


# ---------- Детектор платформы ----------
def detect_platform(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "unknown"

    if any(h in (host or "") for h in ["youtube.com", "youtu.be", "m.youtube.com"]):
        return "youtube"
    return "unknown"

# ---------- Телеграм-бот ----------
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Отправь мне ссылку. Я определю платформу и, если это YouTube, "
        "спаршу комментарии и пришлю Excel-файл.\n\n"
        "Пример: https://www.youtube.com/watch?v=B9oIps6Cb50"
    )

@dp.message(F.text)
async def handle_link(message: Message):
    url = (message.text or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        await message.answer("Пожалуйста, отправь корректную ссылку, начинающуюся с http:// или https://")
        return

    if not YT_API_KEY:
        await message.answer("YT_API_KEY не настроен на сервере. Добавь его в .env и перезапусти бота.")
        return

    await message.answer("Паршу комментарии… Это может занять немного времени.")

    try:
        # Парсинг в отдельном потоке, чтобы не блокировать loop
        import asyncio
        df = await asyncio.to_thread(process_video_comments, url, YT_API_KEY)

        # Готовим Excel в память
        buffer = io.BytesIO()
        # sheet_name короткий, index=False чтобы не мешать
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="comments")
        buffer.seek(0)

        # Имя файла
        safe_name = "youtube_comments.xlsx"
        doc = BufferedInputFile(buffer.read(), filename=safe_name)

        caption = (
            f"Готово ✅\n"
            f"Всего строк: {len(df)}\n"
            f"Столбцы: {', '.join(df.columns)}"
        )
        await message.answer_document(document=doc, caption=caption)

    except ValueError as ve:
        logger.exception("ValueError")
        await message.answer(f"Ошибка: {ve}")
    except Exception as e:
        logger.exception("Unexpected error")
        await message.answer("Произошла непредвиденная ошибка при парсинге. Попробуй другую ссылку или позже.")

# ---------- Запуск ----------
def main():
    import asyncio
    asyncio.run(dp.start_polling(bot))

if __name__ == "__main__":
    main()
