import os
import logging
import cv2
import pytesseract
import tempfile
import re
import asyncio
import aiohttp
import csv
import time
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TESTING_MODE = os.getenv("TESTING_MODE", "False").lower() == "true"

# Logging setup
logging.basicConfig(level=logging.INFO)
user_data = {}

# Ensure data directories exist
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/predictions.csv"):
    with open("data/predictions.csv", "w", newline='') as f:
        csv.writer(f).writerow(["Timestamp", "UserID", "Number", "Prediction", "Confidence", "Validation"])

if not os.path.exists("data/feedback.csv"):
    with open("data/feedback.csv", "w", newline='') as f:
        csv.writer(f).writerow(["Timestamp", "UserID", "Feedback"])

# Extract numbers from image
def extract_numbers_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    numbers = list(map(int, re.findall(r'\b\d+\b', text)))
    return numbers

# Parallel OCR for frames
def ocr_on_frame(frame):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
        cv2.imwrite(img_file.name, frame)
        return extract_numbers_from_image(img_file.name)

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def parallel_ocr_on_frames(frames):
    numbers = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(ocr_on_frame, frames)
        for nums in results:
            numbers += nums
    return numbers

# Prediction logic
def predict_big_small(numbers):
    results = []
    for num in numbers:
        prediction = "Big" if num >= 50 else "Small"
        confidence = 50 + abs(num - 50) / 2
        results.append((num, prediction, round(confidence, 2)))
    return results

def log_prediction(user_id, num, pred, conf, validation):
    with open("data/predictions.csv", "a", newline='') as f:
        csv.writer(f).writerow([datetime.now(), user_id, num, pred, conf, validation])

# GPT validation
async def validate_with_openrouter(number, prediction):
    if TESTING_MODE:
        return "✅ (Testing Mode) Validation Passed"

    prompt = f"एक संख्या {number} है। क्या यह '{prediction}' श्रेणी में आता है? यदि संख्या 50 या उससे अधिक हो तो 'Big' माने, अन्यथा 'Small'।"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.7
    }

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=json_data) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                logging.error(f"OpenRouter API error: {resp.status} {await resp.text()}")
                return f"❌ Validation failed (Status: {resp.status})"

# Handle predictions
async def handle_prediction(numbers, update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    results = predict_big_small(numbers)
    shown = 0

    for num, pred, conf in results[:10]:
        if conf >= 75:
            validation = await validate_with_openrouter(num, pred)
            if any(x in validation.lower() for x in ["सही", "yes", "हाँ", "true"]):
                advice = "🟢 Safe to consider ✅"
                msg = (f"📊 Number: {num}\n📌 Prediction: {pred}\n🎯 Confidence: {conf}%\n🤖 Validation: {validation}\n📢 Advice: {advice}")
                await update.message.reply_text(msg)
                log_prediction(update.effective_chat.id, num, pred, conf, validation)
                shown += 1

    if shown == 0:
        await update.message.reply_text("⚠️ कोई भी मजबूत prediction नहीं मिला। कृपया दूसरी image/video भेजें।")
    else:
        await update.message.reply_text(f"✅ Processed in {round(time.time() - start_time, 2)}s")

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_chat.id] = []
    await update.message.reply_text("✅ Turbo Prediction Bot चालू हो गया है। कृपया image या video भेजें।")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Prediction बंद कर दिया गया है। Memory साफ़ हो गई है।")

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    msg = ' '.join(context.args)
    with open("data/feedback.csv", "a", newline='') as f:
        csv.writer(f).writerow([datetime.now(), user_id, msg])
    await update.message.reply_text("🙏 धन्यवाद! आपका फीडबैक सेव कर लिया गया है।")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    count = 0
    with open("data/predictions.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] == user_id:
                count += 1
    await update.message.reply_text(f"📊 आपने अब तक कुल {count} numbers पर prediction किया है।")

# Media handler
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id

    if update.message.photo:
        file = await update.message.photo[-1].get_file()
    elif update.message.video:
        file = await update.message.video.get_file()
    else:
        await update.message.reply_text("⚠️ कृपया केवल image या video भेजें।")
        return

    suffix = ".jpg" if update.message.photo else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        await file.download_to_drive(f.name)
        file_path = f.name

    if update.message.photo:
        numbers = extract_numbers_from_image(file_path)
    else:
        frames = extract_frames_from_video(file_path)
        step = max(1, len(frames) // 10)
        selected_frames = frames[::step][:10]
        numbers = parallel_ocr_on_frames(selected_frames)

    if user_id not in user_data:
        user_data[user_id] = []

    user_data[user_id] += numbers
    await handle_prediction(numbers, update, context)

# Main
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_media))
    app.run_polling()

if __name__ == "__main__":
    main()
