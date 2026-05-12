import json
import os
import logging
import threading
import numpy as np

from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

import tensorflow as tf

from telegram import Update

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


with open('disease_data.json', 'r') as f:
    disease_db = json.load(f)


CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

MODEL_PATH = 'mycalmilla_plant_model.tflite'  # Fixed: removed /content/ path (Colab-specific)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224


def predict_disease(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])
    preds = preds[0]

    confidence = float(np.max(preds))
    predicted_idx = int(np.argmax(preds))
    label = CLASS_NAMES[predicted_idx]

    if confidence < 0.60:
        return "Unknown Plant", confidence

    return label, confidence


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "🌿 Welcome to Mycalmilla AI Plant Doctor\n\n"
        "I can help identify diseases affecting:\n\n"
        "• Tomato\n"
        "• Potato\n"
        "• Pepper\n\n"
        "Simply send a clear image of a plant leaf.\n\n"
        "You will receive:\n\n"
        "✅ Disease detection\n"
        "✅ Confidence score\n"
        "✅ Organic treatment suggestions\n"
        "✅ Prevention tips\n"
        "✅ Disease explanations"
    )
    await update.message.reply_text(welcome_message)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I'm Mycalmilla AI Plant Doctor!\n"
        "Send me a clear picture of a plant leaf (Tomato, Potato, or Pepper) "
        "and I will try to identify any diseases and provide details."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_path = '/tmp/temp.jpg'
    await file.download_to_drive(image_path)

    await update.message.reply_text("🔎 Mycalmilla AI is analyzing your plant image...")

    try:
        label, confidence = predict_disease(image_path)

        if label == "Unknown Plant":
            await update.message.reply_text(
                "🪴 Sorry, I currently support only:\n\n"
                "• Tomato\n"
                "• Potato\n"
                "• Pepper\n\n"
                "Your uploaded image appears to belong to a crop outside "
                "my current training dataset.\n\n"
                "I'm continually improving and may support this crop "
                "in future updates 😊"
            )
            return

        if confidence >= 0.90:
            confidence_icon = "🟢"
        elif confidence >= 0.75:
            confidence_icon = "🟠"
        else:
            confidence_icon = "🔴"

        disease = disease_db.get(label, {
            "name": label.replace('___', ' - ').replace('__', ' - '),
            "explanation": "No detailed info available.",
            "treatment": [],
            "monitoring_schedule": "Check weekly",
            "prevention_rules": []
        })

        response = (
            f"🌱 *{disease['name']}*\n\n"
            f"{confidence_icon} *Confidence:* {confidence * 100:.1f}%\n\n"
            f"📖 *Explanation:*\n"
            f"{disease['explanation']}\n\n"
        )

        if disease['treatment']:
            response += "*💊 Treatment:*\n"
            response += "\n".join(f"• {t}" for t in disease['treatment'])
            response += "\n\n"

        response += (
            f"🗓️ *Monitoring:*\n"
            f"{disease['monitoring_schedule']}\n\n"
        )

        if disease['prevention_rules']:
            response += "*🛡️ Prevention:*\n"
            response += "\n".join(f"• {p}" for p in disease['prevention_rules'])
            response += "\n\n"

        await update.message.reply_text(response, parse_mode='Markdown')

    except Exception as e:
        logger.error(str(e))
        await update.message.reply_text(
            "❌ Unable to process image.\n\n"
            "Please upload:\n"
            "• a clearer image\n"
            "• good lighting\n"
            "• close-up plant leaf"
        )
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass  # Silence default HTTP logs


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server running on port {port}")
    server.serve_forever()


def main():
    TOKEN = os.environ.get('TELEGRAM_TOKEN')
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable not set. Please set it.")

    # Start health check server in background thread (required for Railway)
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))

    logger.info("Bot is starting...")
    app.run_polling()
    logger.info("Bot stopped.")


if __name__ == '__main__':
    main()
