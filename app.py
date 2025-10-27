import streamlit as st
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import re
from groq import Groq
import os
# ========== CONFIGURATION ==========
MODEL_DIR = "model_llm"
MODEL_FILE = "pytorch_model.bin"
GROQ_API_KEY = st.secrets["YOUR_GROQ_API_KEY"]  # Secure key loading 

# ===========================
# 🧠 CLEANING FUNCTIONS
# ===========================

# Define noise patterns
noise_patterns = [
    r'^\s*\?\s*$',
    r'^\s*\...\s*$',
    r'^\s*\**\s*read title\s*\**\s*$',
    r'^\s*hi\s*$',
    r'^\s*hello\s*$',
    r'^\s*hey\s*$',
    r'^\s*\.\s*$',
    r'^\s*\[deleted by user\]\s*$',
    r'^\s*\\s*$'
]
combined_pattern = re.compile('|'.join(noise_patterns), flags=re.IGNORECASE)


def clean_mental_health_text(text):
    """Clean text content before passing to model."""
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

    # Normalize age/gender expressions
    text = re.sub(r'\[\s*(\d{1,2})\s*[mM]\s*\]', r'\1 year old male', text)
    text = re.sub(r'\[\s*(\d{1,2})\s*[fF]\s*\]', r'\1 year old female', text)
    text = re.sub(r'\(\s*(\d{1,2})\s*[mM]\s*\)', r'\1 year old male', text)
    text = re.sub(r'\(\s*(\d{1,2})\s*[fF]\s*\)', r'\1 year old female', text)
    text = re.sub(r'(\d{1,2})\s*y/o\b', r'\1 year old', text)

    # Expand abbreviations
    shortcuts = {
        r'\bw/': 'with',
        r'\bw/o': 'without',
        r'\babt\b': 'about',
        r'\bbtw\b': 'by the way',
        r'\bidk\b': 'i do not know',
        r'\bomg\b': 'oh my god',
        r'\blol\b': 'laughing',
        r'\bwtf\b': 'what the fuck',
        r'\bomw\b': 'on my way',
        r'\bpls\b': 'please',
        r'\btho\b': 'though',
        r'\bbf\b': 'boyfriend',
        r'\bgf\b': 'girlfriend'
    }
    for pattern, repl in shortcuts.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Medicine-related normalization
    text = re.sub(r'\b(xr|sr)\b', lambda m: m.group(1).upper(), text)
    text = re.sub(r'(\d+)\s*mg\b', r'\1mg', text)

    # Smart $ normalization
    text = re.sub(r'\$\s*(\d+(?:[\.,]\d+)*(?:-\d+(?:[\.,]\d+)*)?)', r'$\1', text)
    text = re.sub(r'(?<!\d)\$(?!\d)', ' ', text)

    # Remove URLs, mentions, markdown links
    text = re.sub(r'http\S+|www\S+|\[.*?\]\(.*?\)', 'weblink', text)
    text = re.sub(r'@\w+', '', text)

    # Remove date formats
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)

    # Normalize mental health labels
    text = re.sub(r'\b([a-zA-Z]*-)?(ocd|adhd|ptsd|bpd|asd)\b',
                  lambda m: m.group(0).lower(), text, flags=re.IGNORECASE)

    # Remove unwanted symbols
    text = re.sub(r'[_*^#<>|\\`~]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'%$\-/]', ' ', text)

    # Collapse extra spaces
    return re.sub(r'\s+', ' ', text).strip().lower()


def is_noisy(text):
    """Check if a text matches known noise patterns."""
    return bool(combined_pattern.match(text.strip()))

# ========== LOAD BERT MODEL ==========
@st.cache_resource
def load_model():
    """Load BERT model and tokenizer from Hugging Face Hub."""
    MODEL_REPO = "Arch11yad/mental-health-bert"

    st.write("🔍 Loading model from Hugging Face Hub...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    st.success("✅ Model loaded successfully from Hugging Face Hub")
    return tokenizer, model, device

tokenizer, model, device = load_model()
# ================== LABEL MAP ==================
label_map = {
    0: "adhd",
    1: "aspergers",
    2: "depression",
    3: "non-suicidal",
    4: "ocd",
    5: "ptsd",
    6: "suicidal",
    7: "mental health"  # fallback for NaN
}

# ================== DETECT FUNCTION ==================
def detect_condition(text: str) -> str:
    """Detect mental health condition from input text."""
    text = text.strip()
    if not text:
        return "Please enter some text."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)  # ✅ move tensors to same device as model

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return label_map.get(pred, "mental health")

# ========== GROQ CLIENT ==========
client = Groq(api_key=GROQ_API_KEY)

# ========== CUSTOM PROMPTS ==========
def generate_prompt(label, user_input):
    base_instruction = (
        "You are a kind, empathetic mental health advisor. "
        "Your goal is to provide emotional relief, reassurance, and self-care advice "
        "to help the person calm down and see hope. "
        "Avoid giving any medical diagnosis or prescriptions."
    )

    condition_prompts = {
        "adhd": "Encourage focus strategies, calm routines, and self-acceptance for occasional distraction.",
        "aspergers": "Be gentle, understanding, and help them embrace their unique perspective without judgment.",
        "depression": "Provide hope, empathy, and gentle reassurance that things can improve. Encourage small positive actions.",
        "non-suicidal": "Offer calm emotional grounding and motivate them to maintain their well-being.",
        "ocd": "Help them reduce obsessive thoughts by promoting calm breathing, mindfulness, and reassurance.",
        "ptsd": "Acknowledge their pain with empathy. Suggest grounding techniques and feeling of safety.",
        "suicidal": "Provide deep emotional support, remove negativity, and encourage reaching out to loved ones or helplines.",
        "mental health": "Offer general comfort and reassurance, helping them reflect and feel supported emotionally."
    }

    return f"""
    {base_instruction}

    The user shows signs of **{label}**.
    {condition_prompts.get(label, "Be supportive and caring.")}

    Their message: "{user_input}"
    """

# ========== ADVICE GENERATION ==========
def generate_advice(label, user_input):
    prompt = generate_prompt(label, user_input)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a compassionate and calming therapist-like AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"⚠️ Could not fetch advice due to an API issue: {e}")
        return "I'm here for you — take a deep breath and remember, help is always available. 💛"

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="🧠 Doctor–Advisor Chatbot", layout="wide")
st.title("🧠 Mental Health Doctor–Advisor Chatbot")
st.markdown("A supportive AI listener that detects your emotions and helps replace negativity with hope 💬")

# =====================
# 🗣️ User Input
# =====================
user_input = st.text_area(
    "How are you feeling today?",
    placeholder="Type how you're feeling...",
    height=120
)

if st.button("Talk with Advisor"):
    if user_input.strip():
        # --- Step 1: Clean text first ---
        clean_text = clean_mental_health_text(user_input)

        # --- Step 2: Handle noisy inputs ---
        if is_noisy(clean_text) or not clean_text:
            st.warning("⚠️ Your input seems too short or unclear — please express a bit more about how you feel 💬")
        else:
            with st.spinner("Analyzing your emotions..."):
                label = detect_condition(clean_text)
                st.success(f"🩺 Detected emotional state: **{label.upper()}**")

            with st.spinner("Responding with empathy..."):
                advice = generate_advice(label, clean_text)
                st.markdown("### 💬 Advisor's Response:")
                st.write(advice)
    else:
        st.warning("Please type something so I can understand how you feel 💬")

st.markdown("---")
st.caption("⚠️ *This chatbot offers emotional support, not medical advice or therapy. "
            "If you're in crisis, please reach out to a mental health professional or local helpline.*")
