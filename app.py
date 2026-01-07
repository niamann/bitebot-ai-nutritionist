import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import random
import google.generativeai as genai
import os
from dotenv import load_dotenv

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="🍎 BiteBot AI Nutritionist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #ff0080, #ff8c00, #ffff00, #00ff00, #00ffff, #0000ff, #8b00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 20px 0;
    }
    .healthy-badge {
        background: linear-gradient(45deg, #00ff88, #00cc66);
        color: #000 !important;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .unhealthy-badge {
        background: linear-gradient(45deg, #ff4444, #ff0066);
        color: #000 !important;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .moderate-badge {
        background: linear-gradient(45deg, #ffcc00, #ffaa00);
        color: #000 !important;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .user-message {
        background: linear-gradient(90deg, rgba(0, 255, 200, 0.15), transparent);
        border-left: 4px solid #00ffcc;
        padding: 12px 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .ai-message {
        background: linear-gradient(90deg, rgba(0, 100, 255, 0.15), transparent);
        border-left: 4px solid #0088ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff0080, #00ccff);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        width: 100%;
    }
    .gemini-message {
        background: linear-gradient(90deg, rgba(147, 51, 234, 0.15), transparent);
        border-left: 4px solid #9333ea;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .gemini-title {
        background: linear-gradient(90deg, #9333ea, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# Session state init
# =========================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "food_log" not in st.session_state: st.session_state.food_log = []
if "ai_chat_history" not in st.session_state: st.session_state.ai_chat_history = []

if "gemini_initialized" not in st.session_state: st.session_state.gemini_initialized = False
if "gemini_error" not in st.session_state: st.session_state.gemini_error = None
if "gemini_model" not in st.session_state: st.session_state.gemini_model = None
if "gemini_chat_session" not in st.session_state: st.session_state.gemini_chat_session = None
if "temp_gemini_key" not in st.session_state: st.session_state.temp_gemini_key = ""

# =========================
# FOOD DATABASE
# =========================
FOOD_DATABASE = {
    "healthy": {
        "apple": 10, "banana": 9, "salad": 10, "broccoli": 10, "spinach": 10,
        "chicken breast": 8, "salmon": 10, "tuna": 9, "eggs": 8, "tofu": 8,
        "greek yogurt": 9, "quinoa": 9, "brown rice": 8, "oats": 9,
        "almonds": 8, "walnuts": 8, "water": 10, "green tea": 9,
        "sushi": 8, "edamame": 9, "grilled fish": 9, "avocado": 9
    },
    "unhealthy": {
        "pizza": 3, "burger": 2, "fries": 1, "fried chicken": 2,
        "donut": 2, "cake": 2, "cookie": 3, "ice cream": 3,
        "chocolate": 4, "candy": 1, "soda": 1, "chips": 1,
        "white bread": 4, "processed meat": 3, "ramen": 3,
        "cheesecake": 2, "cupcake": 2, "milkshake": 2
    },
    "moderate": {
        "pasta": 6, "white rice": 5, "bread": 6, "cheese": 6, "milk": 7,
        "coffee": 6, "juice": 5, "dark chocolate": 7, "red meat": 5,
        "sandwich": 6, "wrap": 6, "soup": 7, "nasi lemak": 6,
        "curry": 6, "satay": 6, "laksa": 5, "biryani": 5, "mee goreng": 4
    }
}

# =========================
# Gemini init + chat (FIXED)
# - Handles missing key with a UI input
# - Auto-picks an available text model via list_models()
# - Only sets gemini_initialized=True after a real test call
# =========================
def _get_api_key():
    load_dotenv()
    # 1) Streamlit secrets (Streamlit Cloud)
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # 2) Environment variable
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY")
    # 3) Manual input (works on any deployment)
    if st.session_state.get("temp_gemini_key"):
        return st.session_state.temp_gemini_key
    return None

def _pick_working_model_name():
    """
    Tries to find a model that supports generateContent.
    Works across different google.generativeai versions (v1beta behaviour).
    """
    try:
        models = genai.list_models()
    except Exception:
        # fallback list if list_models isn't available / fails
        return ["gemini-pro", "models/gemini-pro", "text-bison-001"]

    candidates = []
    for m in models:
        name = getattr(m, "name", "")  # often "models/gemini-pro"
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            candidates.append(name)

    # Put most likely text models first
    preferred = []
    for n in candidates:
        lower = n.lower()
        if "pro" in lower and "vision" not in lower:
            preferred.append(n)
    for n in candidates:
        if n not in preferred:
            preferred.append(n)

    # final fallback if empty
    return preferred if preferred else ["gemini-pro", "models/gemini-pro", "text-bison-001"]

def init_gemini():
    try:
        api_key = _get_api_key()
        if not api_key:
            st.session_state.gemini_initialized = False
            st.session_state.gemini_error = "No GEMINI_API_KEY found. Add it in Streamlit secrets or paste it below."
            st.session_state.gemini_model = None
            st.session_state.gemini_chat_session = None
            return None

        genai.configure(api_key=api_key)

        last_err = None
        model = None
        for model_name in _pick_working_model_name():
            try:
                m = genai.GenerativeModel(model_name)
                _ = m.generate_content("ping")  # ✅ validate
                model = m
                st.session_state.gemini_error = None
                break
            except Exception as e:
                last_err = e

        if not model:
            st.session_state.gemini_initialized = False
            st.session_state.gemini_error = f"Gemini model not usable. Last error: {last_err}"
            st.session_state.gemini_model = None
            st.session_state.gemini_chat_session = None
            return None

        st.session_state.gemini_initialized = True
        st.session_state.gemini_model = model
        return model

    except Exception as e:
        st.session_state.gemini_initialized = False
        st.session_state.gemini_error = str(e)
        st.session_state.gemini_model = None
        st.session_state.gemini_chat_session = None
        return None

class GeminiNutritionAI:
    def __init__(self):
        self.model = None
        self.chat_session = None

    def start_chat(self):
        if not st.session_state.get("gemini_initialized", False) or st.session_state.get("gemini_model") is None:
            self.model = init_gemini()
        else:
            self.model = st.session_state.get("gemini_model")

        if not self.model:
            return False

        if st.session_state.get("gemini_chat_session") is None:
            system_prompt = (
                "You are BiteBot AI Nutritionist, an expert nutritionist and health coach.\n"
                "Guidelines:\n"
                "1. Be friendly, supportive, and non-judgmental\n"
                "2. Provide evidence-based nutrition information\n"
                "3. Give practical, actionable advice\n"
                "4. Consider cultural food preferences\n"
                "5. Use markdown formatting for readability\n"
                "6. Include emojis where appropriate\n"
                "7. Be honest about limitations\n"
            )
            history = [{"role": "user", "parts": [f"SYSTEM:\n{system_prompt}"]}]
            st.session_state.gemini_chat_session = self.model.start_chat(history=history)

        self.chat_session = st.session_state.gemini_chat_session
        return True

    def chat(self, user_message: str) -> str:
        if not self.chat_session:
            if not self.start_chat():
                return "⚠️ Gemini AI not available. Add API key and try again."

        try:
            resp = self.chat_session.send_message(user_message)
            text = (getattr(resp, "text", "") or "").strip()
            return text if text else "⚠️ Gemini returned an empty reply. Try again."
        except Exception as e:
            st.session_state.gemini_error = str(e)
            st.session_state.gemini_initialized = False
            st.session_state.gemini_chat_session = None
            return "⚠️ Gemini AI failed. Using fallback responses."

gemini_ai = GeminiNutritionAI()

# =========================
# Food logic
# =========================
def analyze_food(food_name: str):
    food_lower = food_name.lower()
    for category in ["healthy", "unhealthy", "moderate"]:
        for food, score in FOOD_DATABASE[category].items():
            if food in food_lower:
                if category == "healthy":
                    return {"status": "HEALTHY", "score": score, "badge_class": "healthy-badge",
                            "icon": "✅", "message": "🥗 EXCELLENT! This is super nutritious!", "color": "#00ff88"}
                if category == "unhealthy":
                    return {"status": "UNHEALTHY", "score": score, "badge_class": "unhealthy-badge",
                            "icon": "⚠️", "message": "🔴 Enjoy occasionally in small portions.", "color": "#ff4444"}
                return {"status": "MODERATE", "score": score, "badge_class": "moderate-badge",
                        "icon": "⚖️", "message": "🟡 Good in moderation with balanced diet.", "color": "#ffcc00"}
    return {"status": "MODERATE", "score": random.randint(5, 7), "badge_class": "moderate-badge",
            "icon": "⚖️", "message": "🟡 Enjoy as part of balanced meals.", "color": "#ffcc00"}

def get_nutrition_tips(food: str, status: str):
    food_data = {
        "nasi lemak": {"calories": 600, "protein": 15, "carbs": 75, "fat": 25,
                      "tips": ["Use brown rice for more fiber", "Reduce sambal to lower sodium", "Add boiled egg instead of fried chicken"],
                      "alternatives": ["Nasi kerabu", "Nasi dagang", "Brown rice nasi lemak"]},
        "sandwich": {"calories": 350, "protein": 15, "carbs": 45, "fat": 12,
                     "tips": ["Use whole grain bread", "Load up on vegetables", "Choose lean protein like turkey"],
                     "alternatives": ["Wrap", "Salad bowl", "Open-faced sandwich"]},
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10,
                  "tips": ["Choose thin crust", "Load up on veggies", "Go easy on the cheese"],
                  "alternatives": ["Cauliflower crust pizza", "Veggie pizza", "Whole wheat pizza"]},
        "burger": {"calories": 354, "protein": 20, "carbs": 35, "fat": 15,
                   "tips": ["Use lettuce wrap instead of bun", "Choose lean meat", "Add lots of veggies"],
                   "alternatives": ["Turkey burger", "Veggie burger", "Portobello burger"]},
        "pasta": {"calories": 220, "protein": 8, "carbs": 43, "fat": 1,
                  "tips": ["Choose whole wheat pasta", "Add lean protein", "Load up on vegetables"],
                  "alternatives": ["Zucchini noodles", "Whole wheat pasta", "Lentil pasta"]}
    }
    key = food.lower().strip()
    if key in food_data:
        return food_data[key]

    if status == "HEALTHY":
        return {"calories": random.randint(100, 300), "protein": random.randint(8, 20),
                "carbs": random.randint(10, 30), "fat": random.randint(3, 10),
                "tips": ["Great choice!", "Pair with protein", "Keep up the good work"],
                "alternatives": ["Similar healthy option", "Another good choice", "Variety option"]}
    if status == "UNHEALTHY":
        return {"calories": random.randint(400, 600), "protein": random.randint(5, 15),
                "carbs": random.randint(40, 70), "fat": random.randint(15, 30),
                "tips": ["Enjoy as treat", "Watch portion size", "Balance with veggies"],
                "alternatives": ["Healthier version", "Better alternative", "Light option"]}
    return {"calories": random.randint(250, 400), "protein": random.randint(10, 20),
            "carbs": random.randint(30, 50), "fat": random.randint(8, 20),
            "tips": ["Good in moderation", "Balance your meal", "Enjoy mindfully"],
            "alternatives": ["Healthier twist", "Better choice", "Alternative option"]}

def add_food_to_chat(food: str):
    analysis = analyze_food(food)
    tips = get_nutrition_tips(food, analysis["status"])
    st.session_state.chat_history.append({"type": "user", "content": food, "time": datetime.now().strftime("%H:%M")})
    st.session_state.chat_history.append({"type": "ai", "food": food, "analysis": analysis, "tips": tips, "time": datetime.now().strftime("%H:%M")})
    st.session_state.food_log.append({"food": food, "status": analysis["status"], "score": analysis["score"], "time": datetime.now()})

def display_ai_response(food, analysis, tips):
    with st.container():
        st.markdown(f"""
        <div class='ai-message'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                <div style='font-weight:bold; color:#0088ff;'>🤖 BiteBot AI</div>
                <div style='font-size:0.8rem; color:#aaa;'>{datetime.now().strftime("%H:%M")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### 🍽️ {food.upper()}")
        with c2:
            st.markdown(f"<span class='{analysis['badge_class']}'>{analysis['icon']} {analysis['status']}</span>", unsafe_allow_html=True)

        st.markdown(f'<p style="color:{analysis["color"]}; font-weight:bold; font-size:1.1rem;">{analysis["message"]}</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.1); padding:8px 15px; border-radius:15px; display:inline-block; margin:10px 0;'>
            📊 <b>Nutrition Score:</b> {analysis['score']}/10
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 Nutrition Facts (per serving)")

        cols = st.columns(4)
        items = [("Calories", f"{tips['calories']}", "🔥"),
                 ("Protein", f"{tips['protein']}g", "💪"),
                 ("Carbs", f"{tips['carbs']}g", "🌾"),
                 ("Fat", f"{tips['fat']}g", "🛢️")]
        for i, (label, value, icon) in enumerate(items):
            with cols[i]:
                st.markdown(f"""
                <div style='text-align:center; background:rgba(255,255,255,0.05); padding:15px; border-radius:8px;'>
                    <div style='font-size:1.5rem;'>{icon}</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#00ffcc; margin:5px 0;'>{value}</div>
                    <div style='font-size:0.9rem; color:#aaa;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 💡 Smart Eating Tips")
        for tip in tips["tips"]:
            st.markdown(f"• {tip}")

        st.divider()
        st.markdown("#### 🔄 Healthier Alternatives")
        alt_cols = st.columns(3)
        for i, alt in enumerate(tips["alternatives"]):
            with alt_cols[i]:
                st.info(alt)

        st.divider()
        st.success("🎯 **Pro Advice:** Eat mindfully • Stay hydrated • Enjoy your food • Listen to your body")

def add_ai_chat_message(user_message, ai_response):
    t = datetime.now().strftime("%H:%M")
    st.session_state.ai_chat_history.append({"sender": "user", "message": user_message, "time": t})
    st.session_state.ai_chat_history.append({"sender": "ai", "message": ai_response, "time": t})

# =========================
# UI Header
# =========================
st.markdown('<h1 class="main-title">🍎 BiteBot AI Nutritionist</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#00ffcc;font-size:1.2rem;margin-bottom:30px;">Instant Food Analysis • Smart Nutrition Tips • AI-Powered Chat</p>', unsafe_allow_html=True)

# =========================
# Quick Stats
# =========================
if st.session_state.food_log:
    total = len(st.session_state.food_log)
    healthy = sum(1 for f in st.session_state.food_log if f["status"] == "HEALTHY")
    unhealthy = sum(1 for f in st.session_state.food_log if f["status"] == "UNHEALTHY")
    moderate = total - healthy - unhealthy
    cols = st.columns(4)
    stats = [("🍽️ Total", total, "#00ccff"), ("✅ Healthy", healthy, "#00ff88"),
             ("⚠️ Unhealthy", unhealthy, "#ff4444"), ("⚖️ Moderate", moderate, "#ffcc00")]
    for col, (label, value, color) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{label}</h3>
                <h2 style='color:{color};font-size:2.5rem;'>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["💬 Live Chat", "📊 Food History", "📈 Dashboard", "🤖 AI Chat"])

with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 💬 Live Chat")
        if not st.session_state.chat_history:
            st.info("💬 Start by typing a food or clicking a quick food button!")
        else:
            for msg in st.session_state.chat_history:
                if msg["type"] == "user":
                    st.markdown(f"""
                    <div class='user-message'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;'>
                            <div style='font-weight:bold; color:#00ffcc;'>👤 YOU</div>
                            <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time','')}</div>
                        </div>
                        <div style='font-size:1.1rem;'>🍽️ <b>{msg['content'].upper()}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    display_ai_response(msg["food"], msg["analysis"], msg["tips"])

    with col2:
        st.markdown("### 🍎 Quick Foods")
        quick_foods = [
            ("🍕", "pizza"), ("🥗", "salad"), ("🍔", "burger"), ("🍣", "sushi"),
            ("🍝", "pasta"), ("🍫", "chocolate"), ("🍦", "ice cream"), ("🍎", "apple"),
            ("🥤", "soda"), ("🍗", "chicken"), ("🐟", "fish"), ("🥑", "avocado"),
            ("🍚", "nasi lemak"), ("🍜", "ramen"), ("🥪", "sandwich")
        ]
        for emoji, food in quick_foods:
            if st.button(f"{emoji} {food.title()}", key=f"quick_{food}", use_container_width=True):
                add_food_to_chat(food)
                st.rerun()

    st.markdown("---")
    a, b = st.columns([4, 1])
    with a:
        user_input = st.text_input("", placeholder="e.g., pizza, nasi lemak, burger...", key="food_input")
    with b:
        analyze_btn = st.button("🚀 ANALYZE", use_container_width=True, type="primary")
    if analyze_btn and user_input:
        add_food_to_chat(user_input.lower())
        st.rerun()

with tab2:
    st.markdown("### 📊 Food History")
    if st.session_state.food_log:
        df = pd.DataFrame(st.session_state.food_log).sort_values("time", ascending=False)
        for _, row in df.iterrows():
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{row['food'].title()}**")
                st.caption(f"{row['time'].strftime('%Y-%m-%d %H:%M')}")
            with c2:
                badge = "moderate-badge"
                text = row["status"]
                if row["status"] == "HEALTHY": badge = "healthy-badge"
                if row["status"] == "UNHEALTHY": badge = "unhealthy-badge"
                st.markdown(f'<span class="{badge}" style="font-size:0.8rem; padding:5px 10px;">{text}</span>', unsafe_allow_html=True)
            with c3:
                st.progress(row["score"] / 10, text=f"{row['score']}/10")
            st.divider()
        if st.button("📥 Export History", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "bitebot_history.csv", "text/csv", key="download_csv")
    else:
        st.info("📝 No foods analyzed yet. Start by typing a food above!")

with tab3:
    st.markdown("### 📈 Nutrition Dashboard")
    if st.session_state.food_log:
        df = pd.DataFrame(st.session_state.food_log)
        c1, c2 = st.columns(2)
        with c1:
            status_counts = df["status"].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, title="Food Health Distribution", hole=0.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            food_counts = df["food"].value_counts().head(10)
            fig2 = px.bar(x=food_counts.values, y=food_counts.index, orientation="h", title="Top 10 Foods Analyzed")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                               xaxis_title="Count", yaxis_title="Food")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📈 Analyze some foods to see your nutrition dashboard!")

with tab4:
    st.markdown("### 🤖 Chat with AI Nutritionist")

    # ✅ If no key in secrets/env, allow user to paste it (so Gemini can initialize)
    with st.expander("🔑 Gemini Setup (only if status stays False)"):
        st.text_input("Paste GEMINI_API_KEY here", type="password", key="temp_gemini_key")
        st.caption("If you’re using Streamlit Cloud: set it in App → Settings → Secrets as GEMINI_API_KEY.")

    st.caption(f"Gemini status: {st.session_state.get('gemini_initialized', False)}")
    if st.session_state.get("gemini_error"):
        st.warning(f"Gemini error: {st.session_state.gemini_error}")

    if not st.session_state.get("gemini_initialized", False):
        with st.spinner("🔧 Setting up AI assistant..."):
            init_gemini()

        if st.session_state.get("gemini_initialized", False):
            st.success("✅ Gemini AI initialized successfully!")
        else:
            st.warning("⚠️ Gemini is not initialized. Using fallback responses.")

    fallback_responses = {
        "Give me some healthy meal ideas for weight loss": "🥗 Try: Greek yogurt + berries, chicken salad, salmon + broccoli. Keep protein high + lots of veggies.",
        "What are the best protein sources for muscle building?": "💪 Chicken, eggs, Greek yogurt, salmon, tofu, lentils, chickpeas.",
        "How can I count calories effectively?": "🔥 Track portions, include oils/drinks, be consistent, review weekly trends.",
        "What are common nutrition myths I should know?": "🍎 Myths: carbs/fats aren’t automatically bad, detox cleanses aren’t needed, total intake matters most.",
        "How much water should I drink daily and why?": "💧 Often 2–3L/day (more if active/hot). Helps energy, digestion, performance.",
        "How to create a balanced diet plan?": "📊 Plate method: ½ veg, ¼ protein, ¼ carbs, add healthy fats."
    }

    st.markdown("### 💡 Quick Questions")
    q1, q2 = st.columns(2)

    def ask_ai(user_msg: str):
        if st.session_state.get("gemini_initialized", False):
            resp = gemini_ai.chat(user_msg)
            if resp.startswith("⚠️"):
                resp = fallback_responses.get(user_msg, resp)
        else:
            resp = fallback_responses.get(user_msg, "⚠️ Gemini not available.")
        add_ai_chat_message(user_msg, resp)
        st.rerun()

    with q1:
        if st.button("🥗 Healthy Meal Ideas", use_container_width=True):
            ask_ai("Give me some healthy meal ideas for weight loss")
        if st.button("💪 Protein Sources", use_container_width=True):
            ask_ai("What are the best protein sources for muscle building?")
        if st.button("🔥 Calorie Counting", use_container_width=True):
            ask_ai("How can I count calories effectively?")

    with q2:
        if st.button("🍎 Food Myths", use_container_width=True):
            ask_ai("What are common nutrition myths I should know?")
        if st.button("💧 Hydration Tips", use_container_width=True):
            ask_ai("How much water should I drink daily and why?")
        if st.button("📊 Diet Planning", use_container_width=True):
            ask_ai("How to create a balanced diet plan?")

    st.divider()
    st.markdown("### 💬 Chat History")

    if not st.session_state.ai_chat_history:
        st.info("💭 Ask something or click a quick question!")
    else:
        for msg in st.session_state.ai_chat_history:
            if msg["sender"] == "user":
                st.markdown(f"""
                <div class='user-message'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;'>
                        <div style='font-weight:bold; color:#00ffcc;'>👤 YOU</div>
                        <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time','')}</div>
                    </div>
                    <div style='font-size:1.1rem;'>{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='gemini-message'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                        <div style='font-weight:bold;' class='gemini-title'>🤖 BiteBot AI</div>
                        <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time','')}</div>
                    </div>
                    <div style='font-size:1.1rem;'>{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💭 Ask Your Question")
    i1, i2 = st.columns([4, 1])
    with i1:
        ai_question = st.text_input("", placeholder="e.g., Good snacks for weight loss?", key="ai_question_input")
    with i2:
        ask_btn = st.button("🚀 Ask AI", use_container_width=True, type="primary")

    if ask_btn and ai_question:
        if st.session_state.get("gemini_initialized", False):
            response = gemini_ai.chat(ai_question)
            if response.startswith("⚠️"):
                response = "⚠️ Gemini failed. Please re-check your key or try again."
        else:
            response = "⚠️ Gemini not initialized. Add GEMINI_API_KEY in Streamlit secrets or paste it in the setup expander."
        add_ai_chat_message(ai_question, response)
        st.rerun()

    if st.button("🗑️ Clear AI Chat", use_container_width=True):
        st.session_state.ai_chat_history = []
        st.session_state.gemini_chat_session = None
        st.rerun()

# Clear All
if st.button("🗑️ Clear All History", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.food_log = []
    st.session_state.ai_chat_history = []
    st.session_state.gemini_chat_session = None
    st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;padding:20px;">
    <p>🍎 <b>BiteBot AI Nutritionist</b> • Smart Food Analysis • Real-time Chat • Powered by AI</p>
    <p style="font-size:0.9rem;">Instant food analysis + AI-powered nutrition advice</p>
</div>
""", unsafe_allow_html=True)
