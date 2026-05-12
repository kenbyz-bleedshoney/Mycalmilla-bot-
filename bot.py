import json
import os
import logging
import numpy as np
import time

from PIL import Image

import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from telegram import Update

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# For deployment, consider using os.environ.get('TELEGRAM_TOKEN') instead of userdata.get
# if the environment variable is set directly in your deployment platform (e.g., Render).
# from google.colab import userdata


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

MODEL_PATH = '/content/mycalmilla_plant_model.tflite'

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
    img_array = np.expand_dims(
        img_array,
        axis=0
    )
    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )
    interpreter.invoke()
    preds = interpreter.get_tensor(
        output_details[0]['index']
    )
    preds = preds[0]
    confidence = float(np.max(preds))
    predicted_idx = int(np.argmax(preds))
    label = CLASS_NAMES[predicted_idx]
    if confidence < 0.60:
        return (
            "Unknown Plant",
            confidence
        )
    return label, confidence

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
    """
🌿 Welcome to Mycalmilla AI Plant Doctor

I can help identify diseases affecting:

• Tomato
• Potato
• Pepper

Simply send a clear image of a plant leaf.

You will receive:

✅ Disease detection
✅ Confidence score
✅ Organic treatment suggestions
✅ Prevention tips
✅ Disease explanations
"""
    )
     


async def help_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        """I'm Mycalmilla AI Plant Doctor!
Send me a clear picture of a plant leaf (Tomato, Potato, or Pepper) and I will try to identify any diseases and provide details."""
    )


async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_path = '/tmp/temp.jpg'
    await file.download_to_drive(image_path)

    await update.message.reply_text(
        "🔎 Mycalmilla AI is analyzing your plant image..."
    )

    try:
        label, confidence = predict_disease(image_path)

        if label == "Unknown Plant":
            await update.message.reply_text(
                "🪴 Sorry, I currently support only:

"
                "• Tomato
"
                "• Potato
"
                "• Pepper

"
                "Your uploaded image appears "
                "to belong to a crop outside "
                "my current training dataset.

"
                "I’m continually improving "
                "and may support this crop "
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
            f"🌱 *{disease['name']}*

"
            f"{confidence_icon} "
            f"*Confidence:* "
            f"{confidence*100:.1f}%

"
            f"📖 *Explanation:*
"
            f"{disease['explanation']}

"
        )

        if disease['treatment']:
            response += "*💊 Treatment:*
"
            response += "
".join(
                f"• {t}" for t in disease['treatment']
            )
            response += "

"

        response += (
            f"🗓️ *Monitoring:*
"
            f"{disease['monitoring_schedule']}

"
        )

        if disease['prevention_rules']:
            response += "*🛡️ Prevention:*
"
            response += "
".join(
                f"• {p}" for p in disease['prevention_rules']
            )
            response += "

"

        await update.message.reply_text(
            response,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(str(e))
        await update.message.reply_text(
            "❌ Unable to process image.

"
            "Please upload:
"
            "• a clearer image
"
            "• good lighting
"
            "• close-up plant leaf"
        )
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

def main():
    # For deployment, retrieve the token from environment variables.
    # On Colab, you would typically use `userdata.get('TELEGRAM_TOKEN')`.
    # For Render, you would set 'TELEGRAM_TOKEN' as an environment variable.
    TOKEN = os.environ.get('TELEGRAM_TOKEN')
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable not set. Please set it.")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))

    logger.info("Bot is starting...")
    app.run_polling() # Run the bot in the main thread
    logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
