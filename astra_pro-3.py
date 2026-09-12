# -*- coding: utf-8 -*-
"""
ASTRA AI PRO — Render + Pydroid
Telegram bot + Gemini + Flask web server
"""

import os
import io
import re
import csv
import json
import base64
import zipfile
import logging
import ast
import html as html_lib
import threading

import requests
from flask import Flask

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# ============ ТАНЗИМОТ ============
BOT_TOKEN = os.getenv("ASTRA_BOT_TOKEN", "YOUR_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

HISTORY_FILE = "astra_history.json"
MAX_HISTORY = 100
MAX_TELEGRAM_TEXT = 3900
MAX_FILE_SIZE = 20 * 1024 * 1024

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("ASTRA")

histories = {}

SYSTEM_PROMPT = """
Ту Astra AI Pro ҳастӣ — ёвари касбӣ барои барномасозӣ, веб-сайт,
таҳлили файл ва ҷавобҳои умумӣ.

Қоидаҳо:

1. Ба забони корбар ҷавоб деҳ.
2. Ҷавобҳои дақиқ, сохторнок ва фаҳмо навис.
3. Барои HTML, CSS, JavaScript ва Python коди пурра ва иҷрошаванда деҳ.
4. Кодро дар блокҳои Markdown бо забони дуруст навис.
5. Агар корбар веб-сайт хоҳад, аввал талаботи ӯро фаҳм ва баъд лоиҳаи пурра соз.
6. Ҳеҷ гоҳ API key, password ё token-ро дар frontend нагузор.
7. Агар чизе санҷида нашуда бошад, иддаои "100% бе хато" накун.
8. Барои код шарҳи кӯтоҳи насб ва истифода деҳ.
9. Агар корбар файл хоҳад, матни файлро дар формати тоза навис.
10. Барои саволҳои хатарнок ҷавоби бехатар ва қонунӣ деҳ.
11. Агар корбар пурсад, ки туро кӣ сохт, ҷавоб деҳ:
"Маро Alijon IT сохт. Ман ASTRA AI PRO ҳастам."
"""


# ============ ТАЪРИХ ============
def load_history():
    global histories
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            histories = data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("History load failed: %s", e)
        histories = {}


def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("History save failed: %s", e)


def get_history(chat_id):
    return histories.setdefault(str(chat_id), [])


def add_message(chat_id, role, content):
    history = get_history(chat_id)
    history.append({"role": role, "content": content})
    histories[str(chat_id)] = history[-MAX_HISTORY:]
    save_history()


load_history()


# ============ GEMINI ============
def call_gemini_once(model, messages, extra_parts=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents = []

    for message in messages:
        contents.append({
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}]
        })

    if extra_parts and contents:
        contents[-1]["parts"].extend(extra_parts)

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
            "topP": 0.95
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]
    }

    last_exception = None
    data = None
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                params={"key": GEMINI_KEY},
                json=payload,
                timeout=(20, 180)
            )
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_exception = exc
            logger.warning("Gemini attempt %s/3 failed: %s", attempt + 1, exc)
            if attempt == 2:
                raise RuntimeError(
                    "Пайвастшавӣ ба Gemini қатъ шуд. Интернет/VPN/DNS-ро санҷед."
                ) from exc

    if data is None:
        raise RuntimeError(str(last_exception))

    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Gemini API error"))

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini ҷавоб надод.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if "text" in p)

    if not text:
        raise RuntimeError("Ҷавоб холӣ аст.")

    return text


def call_gemini(messages, extra_parts=None):
    if not GEMINI_KEY or GEMINI_KEY == "YOUR_GEMINI_API_KEY":
        return "❌ GEMINI_API_KEY дар environment гузошта нашудааст."

    last_error = ""
    for model in MODELS:
        try:
            result = call_gemini_once(model, messages, extra_parts)
            logger.info("Gemini model: %s", model)
            return result
        except Exception as e:
            last_error = str(e)
            logger.warning("%s failed: %s", model, e)

    return "❌ ASTRA PRO AI кор накард.\n" + last_error


# ============ ФАЙЛҲО ============
def download_telegram_file(file_id):
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getFile",
            params={"file_id": file_id},
            timeout=30
        )
        result = response.json()
        if not result.get("ok"):
            return None

        file_path = result["result"]["file_path"]
        response = requests.get(
            f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}",
            timeout=120
        )
        return response.content if response.status_code == 200 else None
    except Exception as e:
        logger.error("Download failed: %s", e)
        return None


def b64(data):
    return base64.b64encode(data).decode("utf-8")


def decode_text(data):
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


async def send_txt(message, text, filename="astra.txt"):
    bio = io.BytesIO(text.encode("utf-8"))
    bio.seek(0)
    await message.reply_document(document=InputFile(bio, filename=filename))


async def send_csv(message, rows, filename="astra.csv"):
    sio = io.StringIO()
    csv.writer(sio).writerows(rows)
    bio = io.BytesIO(sio.getvalue().encode("utf-8-sig"))
    bio.seek(0)
    await message.reply_document(document=InputFile(bio, filename=filename))


async def send_pdf(message, text, filename="astra.pdf"):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
    except ImportError:
        await message.reply_text("reportlab насб нест:\npip install reportlab")
        return

    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for paragraph in text.split("\n\n"):
        safe = html_lib.escape(paragraph).replace("\n", "<br/>")
        story.append(Paragraph(safe, styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    bio.seek(0)
    await message.reply_document(document=InputFile(bio, filename=filename))


async def send_docx(message, text, filename="astra.docx"):
    try:
        from docx import Document
    except ImportError:
        await message.reply_text("python-docx насб нест:\npip install python-docx")
        return

    doc = Document()
    for paragraph in text.split("\n\n"):
        doc.add_paragraph(paragraph)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    await message.reply_document(document=InputFile(bio, filename=filename))


async def send_zip(message, files, filename="astra_project.zip"):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    bio.seek(0)
    await message.reply_document(document=InputFile(bio, filename=filename))


# ============ САНҶИШИ КОД ============
def extract_code_blocks(text):
    pattern = r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)```"
    return re.findall(pattern, text, flags=re.DOTALL)


def validate_code(language, code):
    language = language.lower().strip()

    if language in ("python", "py"):
        try:
            ast.parse(code)
            return True, "Python syntax OK"
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"

    if language == "json":
        try:
            json.loads(code)
            return True, "JSON syntax OK"
        except Exception as e:
            return False, f"JSON error: {e}"

    if language in ("html", "htm"):
        return True, "HTML found; manual review recommended"

    if language in ("css", "javascript", "js"):
        if code.count("{") != code.count("}"):
            return False, f"{language} braces mismatch"
        return True, f"{language} basic check OK"

    return True, "No validator for this language"


def validate_response(text):
    results = []
    for language, code in extract_code_blocks(text):
        ok, detail = validate_code(language, code)
        results.append({"language": language or "text", "ok": ok, "detail": detail})
    return results


def validation_report(text):
    results = validate_response(text)
    if not results:
        return "ℹ️ Блоки коди Markdown ёфт нашуд."

    lines = ["🔍 Санҷиши код:"]
    for item in results:
        icon = "✅" if item["ok"] else "❌"
        lines.append(f"{icon} {item['language']}: {item['detail']}")
    return "\n".join(lines)


def safe_filename(name):
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name[:100] or "file.txt"


def extract_project_files(text):
    pattern = r"(?:FILE|FILE_NAME|ФАЙЛ)\s*:\s*([^\n]+)\n\s*```[^\n]*\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    files = {}

    for name, content in matches:
        files[safe_filename(name.strip())] = content.strip() + "\n"
    return files


def make_project_files(text):
    files = extract_project_files(text)
    if files:
        return files

    extension_map = {
        "html": "index.html",
        "css": "style.css",
        "javascript": "script.js",
        "js": "script.js",
        "python": "main.py",
        "py": "main.py",
        "json": "data.json",
        "": "code.txt"
    }

    for language, code in extract_code_blocks(text):
        name = extension_map.get(language.lower(), "code.txt")
        if name not in files:
            files[name] = code

    return files


# ============ HANDLERS ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    histories[chat_id] = []
    save_history()

    await update.message.reply_text(
        "🚀 Astra AI Pro фаъол шуд!\n\n"
        "Ман метавонам:\n"
        "• Код ва веб-сайт созам\n"
        "• Файлҳои TXT, PDF, DOCX, CSV ва ZIP созам\n"
        "• Файлҳо ва расмҳоро таҳлил кунам\n"
        "• Таърихи сӯҳбатро нигоҳ дорам\n"
        "• Кодро санҷиши ибтидоӣ кунам\n\n"
        "Фармонҳо:\n"
        "/help — кӯмак\n"
        "/clear — тоза кардани таърих\n"
        "/txt матн — TXT\n"
        "/pdf матн — PDF\n"
        "/csv — CSV\n"
        "/zip — ZIP\n"
        "/project — сохтани ZIP аз ҷавоби охирин"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Кӯмаки Astra AI Pro\n\n"
        "Матн навис: ҷавоб мегирӣ.\n"
        "Расм ё файл фирист: таҳлил мешавад.\n\n"
        "Барои сохтани веб-сайт:\n"
        "«Барои ман веб-сайти пурра соз. HTML, CSS ва JS-ро "
        "дар файлҳои алоҳида навис. Номҳои файлро бо FILE: нишон деҳ.»\n\n"
        "Барои ZIP:\n"
        "Ба ҷавоби веб-сайт /project фирист.\n\n"
        "Фармонҳо:\n"
        "/start /help /clear /txt /pdf /csv /zip /project"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    histories[chat_id] = []
    save_history()
    await update.message.reply_text("🗑 Таърих тоза шуд.")


async def txt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_txt(update.message, " ".join(context.args) or "Холӣ")


async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_pdf(update.message, " ".join(context.args) or "Холӣ")


async def csv_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = [
        ["Ном", "Синну сол", "Шаҳр"],
        ["Али", "15", "Душанбе"],
        ["Зарина", "16", "Хуҷанд"],
    ]
    await send_csv(update.message, rows)


async def zip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    files = {
        "README.txt": "Astra AI Pro\n",
        "info.json": json.dumps(
            {"bot": "Astra", "version": "Pro"},
            ensure_ascii=False, indent=2
        )
    }
    await send_zip(update.message, files)


async def project_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    history = histories.get(chat_id, [])
    last_reply = next(
        (item["content"] for item in reversed(history) if item["role"] == "assistant"),
        ""
    )
    files = make_project_files(last_reply)

    if not files:
        await update.message.reply_text("❌ Дар ҷавоби охирин код ё файл ёфт нашуд.")
        return

    await send_zip(update.message, files, "astra_project.zip")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    chat_id = str(update.effective_chat.id)
    user_text = msg.text or msg.caption or ""
    extra_parts = []

    if msg.photo:
        data = download_telegram_file(msg.photo[-1].file_id)
        if data:
            extra_parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": b64(data)}
            })
        user_text = user_text or "Ин расмро таҳлил кун."

    elif msg.document:
        doc = msg.document
        if doc.file_size and doc.file_size > MAX_FILE_SIZE:
            await msg.reply_text("❌ Файл аз ҳад калон аст.")
            return

        data = download_telegram_file(doc.file_id)
        if data:
            mime = doc.mime_type or "application/octet-stream"
            text_data = decode_text(data)

            if mime.startswith("image/"):
                extra_parts.append({
                    "inline_data": {"mime_type": mime, "data": b64(data)}
                })
            elif text_data is not None:
                user_text = user_text or f"Файли {doc.file_name}-ро таҳлил кун."
                extra_parts.append({
                    "text": "\n\n[FILE CONTENT]\n" + text_data[:30000]
                })
            else:
                user_text = user_text or f"Файли {doc.file_name}-ро таҳлил кун."

    elif msg.video:
        if msg.video.file_size and msg.video.file_size > MAX_FILE_SIZE:
            await msg.reply_text("❌ Видео аз ҳад калон аст.")
            return

        data = download_telegram_file(msg.video.file_id)
        if data:
            extra_parts.append({
                "inline_data": {
                    "mime_type": msg.video.mime_type or "video/mp4",
                    "data": b64(data)
                }
            })
        user_text = user_text or "Ин видеоро таҳлил кун."

    elif msg.voice or msg.audio:
        obj = msg.voice or msg.audio
        data = download_telegram_file(obj.file_id)
        if data:
            extra_parts.append({
                "inline_data": {
                    "mime_type": obj.mime_type or "audio/ogg",
                    "data": b64(data)
                }
            })
        user_text = user_text or "Ин аудиоро таҳлил кун."

    if not user_text and not extra_parts:
        await msg.reply_text("Матн, расм ё файл фирист.")
        return

    add_message(chat_id, "user", user_text)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass

    reply = call_gemini(get_history(chat_id), extra_parts=extra_parts)
    add_message(chat_id, "assistant", reply)

    project_words = ["веб-сайт", "website", "html", "css", "javascript", "сайт соз", "код соз"]
    if any(word in user_text.lower() for word in project_words):
        files = make_project_files(reply)
        if files:
            await send_zip(msg, files, "astra_project.zip")

    if extract_code_blocks(reply):
        await msg.reply_text(validation_report(reply))

    for start_index in range(0, len(reply), MAX_TELEGRAM_TEXT):
        await msg.reply_text(reply[start_index:start_index + MAX_TELEGRAM_TEXT])


# ============ FLASK WEB SERVER (барои Render) ============
web_app = Flask(__name__)


@web_app.route("/")
def index():
    return "Astra AI Pro — running", 200


@web_app.route("/health")
def health():
    return {"status": "ok", "bot": "Astra AI Pro"}, 200


def run_web():
    port = int(os.getenv("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ============ MAIN ============
def start_bot():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("txt", txt_command))
    application.add_handler(CommandHandler("pdf", pdf_command))
    application.add_handler(CommandHandler("csv", csv_command))
    application.add_handler(CommandHandler("zip", zip_command))
    application.add_handler(CommandHandler("project", project_command))

    application.add_handler(
        MessageHandler(
            filters.TEXT | filters.PHOTO | filters.Document.ALL |
            filters.VIDEO | filters.VOICE | filters.AUDIO,
            handle_message
        )
    )

    logger.info("Бот омода аст. Telegram-ро кушо ва /start фирист.")
    application.run_polling(drop_pending_updates=True)


def main():
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.error("❌ ASTRA_BOT_TOKEN дар environment гузор.")
        return

    logger.info("🚀 Astra AI Pro started")
    logger.info("🤖 Сохтаи Alijon IT")

    # Web server дар thread алоҳида — барои Render
    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()

    # Бот дар thread асосӣ
    start_bot()


if __name__ == "__main__":
    main()
