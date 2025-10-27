import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from groq import Groq
import os
# ========== CONFIGURATION ==========
MODEL_DIR = "model_llm"
GROQ_API_KEY = st.secrets["YOUR_GROQ_API_KEY"]  # Secure key loading 

# ========== LOAD BERT MODEL ==========
@st.cache_resource
def load_model():
    st.write("🔍 Model folder contents:", os.listdir(MODEL_DIR))

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    try:
        # Try normal load first
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        st.write("✅ Model loaded successfully from pytorch_model.bin")
    except Exception as e:
        st.error(f"⚠️ Direct model load failed: {e}")
        model = BertForSequenceClassification.from_pretrained(
            MODEL_DIR, ignore_mismatched_sizes=True
        )
        weight_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            st.write("✅ Loaded weights manually from pytorch_model.bin")
        else:
            st.error("❌ No pytorch_model.bin found!")

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping (8th is NaN → handled as 'mental health')
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

# ========== DETECT CONDITION ==========
def detect_condition(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(pred, "mental health")  # safe fallback

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

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a compassionate and calming therapist-like AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="🧠 Doctor–Advisor Chatbot", layout="wide")
st.title("🧠 Mental Health Doctor–Advisor Chatbot")
st.markdown("A supportive AI listener that detects your emotions and helps replace negativity with hope 💬")

user_input = st.text_area("How are you feeling today?", placeholder="Type how you're feeling...", height=120)

if st.button("Talk with Advisor"):
    if user_input.strip():
        with st.spinner("Analyzing your emotions..."):
            label = detect_condition(user_input)
            st.success(f"🩺 Detected emotional state: **{label.upper()}**")

        with st.spinner("Responding with empathy..."):
            advice = generate_advice(label, user_input)
            st.markdown(f"### 💬 Advisor's Response:")
            st.write(advice)
    else:
        st.warning("Please type something so I can understand how you feel 💬")

st.markdown("---")
st.caption("⚠️ *This chatbot offers emotional support, not medical advice or therapy. "
            "If you're in crisis, please reach out to a mental health professional or local helpline.*")
