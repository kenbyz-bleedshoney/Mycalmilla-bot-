# -*- coding: utf-8 -*-
pip install --upgrade google-generativeai
import json
import os
import logging
import threading
import numpy as np
import google.generativeai as genai

from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
except ImportError:
    from tflite_litert.interpreter import Interpreter as LiteInterpreter

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load Disease Database ─────────────────────────────────────
with open('disease_data.json', 'r') as f:
    disease_db = json.load(f)

# ── Load Custom Knowledge ─────────────────────────────────────
KNOWLEDGE_FILE = 'custom_knowledge.json'
if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, 'r') as f:
        custom_knowledge = json.load(f)
else:
    custom_knowledge = {}

def save_knowledge():
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump(custom_knowledge, f, indent=2)

# ── Build Rich Knowledge Base for Gemini ──────────────────────
def build_knowledge_base() -> str:
    """Converts disease_db + custom_knowledge into a rich text block for Gemini."""
    kb = ""
    for disease_key, info in disease_db.items():
        name = disease_key.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')
        kb += f"\n--- {name} ---\n"
        kb += f"Explanation: {info.get('explanation', '')}\n"
        kb += f"Treatment: {info.get('treatment', '')}\n"
        kb += f"Monitoring: {info.get('monitoring', '')}\n"
        kb += f"Prevention: {info.get('prevention', '')}\n"
        if disease_key in custom_knowledge:
            kb += f"Additional Notes: {custom_knowledge[disease_key]}\n"
    return kb

# ── Gemini System Prompt ───────────────────────────────────────
GEMINI_SYSTEM_PROMPT = """
You are Mycalmilla, a world-class AI plant doctor and agricultural expert assistant on Telegram.

YOUR EXPERTISE COVERS:
- Plant disease diagnosis and treatment (fungal, bacterial, viral, pest-related)
- Organic and chemical treatment recommendations with specific product names
- Crop management for Tomato, Potato, Pepper and all other crops
- Soil health, fertilization, irrigation advice
- Pest control strategies
- Seasonal farming advice
- Post-harvest handling
- General agriculture and farming best practices

YOUR PERSONALITY:
- Warm, friendly, and encouraging to farmers
- Practical and specific — always give actionable advice
- Mention both organic AND chemical options when relevant
- Use simple language farmers can understand
- Be concise but thorough

YOUR DISEASE KNOWLEDGE BASE:
{knowledge_base}

IMPORTANT RULES:
- Always give specific chemical or organic product names when asked (e.g. "copper hydroxide", "mancozeb", "neem oil")
- If asked about a disease not in your knowledge base, use your general agricultural expertise to answer
- If asked non-farming questions, politely redirect to farming topics
- After a plant scan, remember the detected disease and use it as context for follow-up questions
- Never say you cannot help with farming questions — always try your best
- Keep responses under 300 words unless the question needs more detail
"""

# ── Gemini Setup ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
gemini_model = None

def setup_gemini():
    global gemini_model
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        knowledge_base = build_knowledge_base()
        system_prompt = GEMINI_SYSTEM_PROMPT.format(knowledge_base=knowledge_base)
        gemini_model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-002'
            system_instruction=system_prompt
        )
        logger.info("Gemini AI configured successfully")
    else:
        logger.warning("GEMINI_API_KEY not set - conversational AI disabled")

setup_gemini()

# ── Class Names ───────────────────────────────────────────────
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

# ── Model Setup ───────────────────────────────────────────────
MODEL_PATH = 'mycalmilla_plant_model.tflite'
interpreter = LiteInterpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = 224

# ── Per-user session memory ───────────────────────────────────
user_sessions = {}

# ── Admin ID ──────────────────────────────────────────────────
ADMIN_ID = int(os.environ.get('ADMIN_TELEGRAM_ID', '0'))

# ── Intent Keywords ───────────────────────────────────────────
GREETINGS = ['good morning', 'good afternoon', 'good evening', 'good night',
             'hello', 'hi', 'hey', 'howdy', 'greetings', 'good day']
THANKS = ['thank you', 'thanks', 'thank u', 'thx', 'appreciated', 'grateful', 'well done']
DISSATISFACTION = ['wrong', 'incorrect', 'not right', 'bad result', 'mistake',
                   'inaccurate', 'disappointed', 'useless', 'not working']
COMPLIMENTS = ['great', 'excellent', 'amazing', 'awesome', 'good job',
               'fantastic', 'perfect', 'wonderful', 'brilliant']


# ── Disease Prediction ────────────────────────────────────────
def predict_disease(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224)) # MobileNet default size
    img_array = np.array(img).astype(np.float32)
    
    # CRITICAL: Normalize your pixels!
    img_array = img_array / 255.0 
    
    img_array = np.expand_dims(img_array, axis=0)
    # ... rest of your prediction logic
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    confidence = float(np.max(preds))
    predicted_idx = int(np.argmax(preds))
    label = CLASS_NAMES[predicted_idx]

    if confidence < 0.60:
        return "Unknown Plant", confidence

    return label, confidence


# ── Gemini Chat ───────────────────────────────────────────────
async def ask_gemini(user_message: str, last_disease: str = None) -> str:
    if not gemini_model:
        return (
            "I can answer farming questions but my AI brain isn't connected right now.\n"
            "Please send a leaf photo and I'll analyze it! \U0001f331"
        )

    # Add last detected disease as extra context if available
    context_prefix = ""
    if last_disease:
        disease_name = last_disease.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')
        context_prefix = f"[Context: The user's last plant scan detected '{disease_name}']\n\n"

    try:
        response = gemini_model.generate_content(context_prefix + user_message)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "I had trouble thinking right now. Please try again! \U0001f331"


# ── Intent Detection ──────────────────────────────────────────
def detect_intent(text: str):
    t = text.lower()
    if any(g in t for g in GREETINGS):
        return 'greeting'
    if any(x in t for x in THANKS):
        return 'thanks'
    if any(x in t for x in DISSATISFACTION):
        return 'dissatisfaction'
    if any(x in t for x in COMPLIMENTS):
        return 'compliment'
    return 'question'


# ── Command Handlers ──────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f33f Welcome to Mycalmilla AI Plant Doctor\n\n"
        "I can help you with:\n\n"
        "\U0001f4f8 Plant disease detection \u2014 just send a leaf photo\n"
        "\U0001f4ac Farming advice \u2014 ask me anything about crops\n"
        "\U0001f48a Treatment options \u2014 organic and chemical\n"
        "\U0001f6e1 Prevention tips and crop management\n\n"
        "Supported crops for image scanning:\n"
        "\u2022 Tomato \u2022 Potato \u2022 Pepper\n\n"
        "But I can answer farming questions about ANY crop! \U0001f331"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f33f Mycalmilla AI Plant Doctor\n\n"
        "What I can do:\n\n"
        "\U0001f4f8 Send a leaf photo \u2192 disease diagnosis\n"
        "\U0001f4ac Ask follow-up questions after a scan\n"
        "\U0001f9ea Ask about chemicals, treatments, fertilizers\n"
        "\U0001f331 Ask about any crop disease or farming topic\n"
        "\U0001f44b Greet me, thank me \u2014 I respond!\n\n"
        "Example questions:\n"
        "\u2022 What chemical can I use for tomato late blight?\n"
        "\u2022 How do I prevent early blight?\n"
        "\u2022 What fertilizer is good for pepper?\n"
        "\u2022 How often should I water my tomato plants?"
    )


async def teach_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ADMIN_ID == 0 or user_id != ADMIN_ID:
        await update.message.reply_text("\u274c You are not authorized to use this command.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /teach DiseaseName: additional information\n\n"
            "Example:\n"
            "/teach Tomato_Late_blight: Ridomil Gold and Infinito are effective systemic fungicides. "
            "Apply every 7 days during rainy season."
        )
        return

    full_text = ' '.join(context.args)
    if ':' not in full_text:
        await update.message.reply_text("Please use format: /teach DiseaseName: information")
        return

    disease_key, info = full_text.split(':', 1)
    disease_key = disease_key.strip()
    info = info.strip()

    existing = custom_knowledge.get(disease_key, "")
    custom_knowledge[disease_key] = (existing + " " + info).strip()
    save_knowledge()

    # Rebuild Gemini with updated knowledge
    setup_gemini()

    await update.message.reply_text(
        f"\u2705 Knowledge updated!\n\n"
        f"Topic: {disease_key}\n"
        f"Added: {info}\n\n"
        f"Gemini has been updated with this new knowledge."
    )


async def knowledge_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ADMIN_ID == 0 or user_id != ADMIN_ID:
        await update.message.reply_text("\u274c You are not authorized.")
        return

    if not custom_knowledge:
        await update.message.reply_text("No custom knowledge stored yet.\n\nUse /teach to add knowledge.")
        return

    msg = "\U0001f4da Custom Knowledge Base:\n\n"
    for k, v in custom_knowledge.items():
        msg += f"\U0001f538 {k}:\n{v}\n\n"
    await update.message.reply_text(msg)


# ── Photo Handler ─────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_path = f'/tmp/plant_{user_id}.jpg'
    await file.download_to_drive(image_path)

    await update.message.reply_text("\U0001f50e Mycalmilla AI is analyzing your plant image...")

    try:
        label, confidence = predict_disease(image_path)

        if label == "Unknown Plant":
            await update.message.reply_text(
                "\U0001fab4 I couldn't confidently identify this plant.\n\n"
                "This could mean:\n"
                "\u2022 The crop is outside my training (I scan Tomato, Potato, Pepper)\n"
                "\u2022 The image quality is too low\n\n"
                "Tips for better results:\n"
                "\u2022 Close-up shot of a single leaf\n"
                "\u2022 Good natural lighting\n"
                "\u2022 Leaf flat and in focus\n\n"
                "You can still ask me text questions about any plant disease! \U0001f331"
            )
            return

        # Store session
        user_sessions[user_id] = {
            'last_disease': label,
            'confidence': confidence
        }

        if confidence >= 0.90:
            confidence_icon = "\U0001f7e2"
        elif confidence >= 0.75:
            confidence_icon = "\U0001f7e0"
        else:
            confidence_icon = "\U0001f534"

        disease = disease_db.get(label) or {
            "explanation": "No detailed info available.",
            "treatment": "No treatment info available.",
            "monitoring": "Check weekly.",
            "prevention": "No prevention info available."
        }

        custom_info = custom_knowledge.get(label, "")
        disease_name = label.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')

        response = (
            f"\U0001f331 *{disease_name}*\n\n"
            f"{confidence_icon} *Confidence:* {confidence * 100:.1f}%\n\n"
            f"\U0001f4d6 *Explanation:*\n{disease.get('explanation', 'N/A')}\n\n"
            f"*\U0001f48a Treatment:*\n{disease.get('treatment', 'N/A')}\n\n"
            f"\U0001f5d3 *Monitoring:*\n{disease.get('monitoring', 'Check weekly')}\n\n"
            f"*\U0001f6e1 Prevention:*\n{disease.get('prevention', 'N/A')}\n\n"
        )

        if custom_info:
            response += f"*\U0001f4cc Additional Notes:*\n{custom_info}\n\n"

        response += "_You can ask me follow-up questions about this disease! \U0001f331_"

        await update.message.reply_text(response, parse_mode='Markdown')

    except Exception as e:
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")
        await update.message.reply_text(
            "\u274c Unable to process image.\n\n"
            "Please upload:\n"
            "\u2022 a clearer image\n"
            "\u2022 good lighting\n"
            "\u2022 close-up plant leaf"
        )
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# ── Text Message Handler ──────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    intent = detect_intent(text)
    last_disease = user_sessions.get(user_id, {}).get('last_disease')

    if intent == 'greeting':
        await update.message.reply_text(
            "\U0001f33f Hello! I'm Mycalmilla, your AI Plant Doctor.\n\n"
            "Send me a leaf photo for disease scanning, or ask me any farming question!"
        )
    elif intent == 'thanks':
        await update.message.reply_text(
            "\U0001f60a You're welcome! Happy to help keep your crops healthy.\n\n"
            "Feel free to ask me anything about farming anytime! \U0001f331"
        )
    elif intent == 'dissatisfaction':
        await update.message.reply_text(
            "\U0001f64f I'm sorry the result wasn't accurate!\n\n"
            "For better image scans:\n"
            "\u2022 Close-up, well-lit photo\n"
            "\u2022 Single leaf in focus\n"
            "\u2022 Leaf flat in frame\n\n"
            "You can also describe the symptoms in text and I'll try to help! \U0001f331"
        )
    elif intent == 'compliment':
        await update.message.reply_text(
            "\U0001f60a Thank you! That means a lot.\n\n"
            "I'm always here to help protect your crops! \U0001f33f"
        )
    else:
        # All questions — farming, chemicals, treatment, general — go to Gemini
        await update.message.chat.send_action('typing')
        response = await ask_gemini(text, last_disease)
        await update.message.reply_text(response)


# ── Health Check Server ───────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server running on port {port}")
    server.serve_forever()


# ── Main ──────────────────────────────────────────────────────
def main():
    TOKEN = os.environ.get('TELEGRAM_TOKEN')
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable not set.")

    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("teach", teach_cmd))
    app.add_handler(CommandHandler("knowledge", knowledge_cmd))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Mycalmilla Bot is starting...")
    app.run_polling(drop_pending_updates=True)
    logger.info("Bot stopped.")


if __name__ == '__main__':
    main()
