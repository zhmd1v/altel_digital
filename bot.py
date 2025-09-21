# bot.py
#hello
import os
import io
import logging
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from get_response import process_video_comments
from matplotlib import pyplot as plt

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

def visualize_comments(df: pd.DataFrame):
    """
    Ожидает колонки: desired (bool), type (str), tone (str).
    Рендерит три картинки: desired.png, type.png, tone.png.
    Возвращает словарь со статистикой.
    """
    stats = {}

    # safety: если колонок нет — не падаем
    required = {"desired", "type", "tone"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"В DataFrame отсутствуют колонки: {', '.join(missing)}")

    # Общее количество комментариев
    total_comments = len(df)
    stats["total_comments"] = total_comments

    # Распределение по желательности
    desired_counts = df["desired"].value_counts(dropna=False)

    # Тип (только желательные)
    only_desired = df[df["desired"] == True]
    type_counts = only_desired["type"].value_counts()
    if type_counts.sum() > 0:
        type_percent = type_counts / type_counts.sum() * 100
        stats["type_distribution"] = {
            t: {"count": int(type_counts[t]), "percent": round(type_percent[t], 1)}
            for t in type_counts.index
        }
    else:
        stats["type_distribution"] = {}

    # Тональность (только желательные)
    tone_counts = only_desired["tone"].value_counts()
    if tone_counts.sum() > 0:
        tone_percent = tone_counts / tone_counts.sum() * 100
        stats["tone_distribution"] = {
            t: {"count": int(tone_counts[t]), "percent": round(tone_percent[t], 1)}
            for t in tone_counts.index
        }
    else:
        stats["tone_distribution"] = {}

    # 1) Pie: желательные/нежелательные
    plt.figure(figsize=(6, 6))
    labels_map = {True: "Желательные", False: "Нежелательные"}
    labels = [labels_map.get(idx, str(idx)) for idx in desired_counts.index]
    plt.pie(
        desired_counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#66b3ff", "#ff9999"]  # можно убрать/поменять
    )
    plt.title("Распределение комментариев по желательности")
    plt.savefig("desired.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Pie: тип (только желательные)
    if len(type_counts) > 0:
        plt.figure(figsize=(6, 6))
        plt.pie(
            type_counts,
            labels=type_counts.index,
            autopct=lambda p: f"{int(p * type_counts.sum() / 100)} ({p:.1f}%)",
            startangle=140
        )
        plt.title("Распределение комментариев по типу (только желательные)")
        plt.savefig("type.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        # пустую заглушку, чтобы код отправки был единообразным
        plt.figure(figsize=(6, 3))
        plt.title("Нет желательных комментариев для диаграммы по типам")
        plt.savefig("type.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3) Pie: тональность (только желательные)
    if len(tone_counts) > 0:
        plt.figure(figsize=(6, 6))
        plt.pie(
            tone_counts,
            labels=tone_counts.index,
            autopct=lambda p: f"{int(p * tone_counts.sum() / 100)} ({p:.1f}%)",
            startangle=140
        )
        plt.title("Распределение комментариев по тональности (только желательные)")
        plt.savefig("tone.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(6, 3))
        plt.title("Нет желательных комментариев для диаграммы по тональности")
        plt.savefig("tone.png", dpi=150, bbox_inches="tight")
        plt.close()

    return stats


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
        stats = await asyncio.to_thread(visualize_comments, df)

        # Готовим Excel в память
        buffer = io.BytesIO()
        # sheet_name короткий, index=False чтобы не мешать
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="comments")
        buffer.seek(0)

        # Имя файла
        safe_name = "youtube_comments.xlsx"
        doc = BufferedInputFile(buffer.read(), filename=safe_name)

        with open("desired.png", "rb") as f:
            desired_png = BufferedInputFile(f.read(), filename="desired.png")
        with open("type.png", "rb") as f:
            type_png = BufferedInputFile(f.read(), filename="type.png")
        with open("tone.png", "rb") as f:
            tone_png = BufferedInputFile(f.read(), filename="tone.png")

        caption = (
            f"Готово ✅\n"
            f"Всего строк: {len(df)}\n"
            f"Столбцы: {', '.join(df.columns)}"
        )
        await message.answer_document(document=doc, caption=caption)
        await message.answer_photo(desired_png, caption="Распределение по желательности")
        await message.answer_photo(type_png, caption="Типы (только желательные)")
        await message.answer_photo(tone_png, caption="Тональность (только желательные)")

        def fmt_dist(d):
            if not d:
                return "—"
            return "\n".join(f"• {k}: {v['count']} ({v['percent']}%)" for k, v in d.items())

        summary = (
            f"Итоги:\n"
            f"• Всего комментариев: {stats.get('total_comments', 0)}\n\n"
            f"Тип (желательные):\n{fmt_dist(stats.get('type_distribution', {}))}\n\n"
            f"Тональность (желательные):\n{fmt_dist(stats.get('tone_distribution', {}))}"
        )
        await message.answer(summary)


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
