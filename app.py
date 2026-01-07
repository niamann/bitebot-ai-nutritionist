import streamlit as st
import pandas as pd
import numpy as np
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "food_log" not in st.session_state:
    st.session_state.food_log = []
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

if "gemini_initialized" not in st.session_state:
    st.session_state.gemini_initialized = False
if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = None
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "gemini_chat_session" not in st.session_state:
    st.session_state.gemini_chat_session = None

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
# Gemini init + chat
# =========================
def init_gemini():
    """
    Initialize Gemini using google.generativeai
    Works on Streamlit Cloud via st.secrets or locally via .env / env var.
    """
    try:
        load_dotenv()

        api_key = None
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.session_state.gemini_initialized = False
            st.session_state.gemini_error = "GEMINI_API_KEY not found (Streamlit secrets or environment)."
            st.session_state.gemini_model = None
            st.session_state.gemini_chat_session = None
            return None

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        st.session_state.gemini_initialized = True
        st.session_state.gemini_error = None
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

    def ensure_ready(self) -> bool:
        # if not initialized, try init
        if not st.session_state.get("gemini_initialized", False) or st.session_state.get("gemini_model") is None:
            self.model = init_gemini()
        else:
            self.model = st.session_state.get("gemini_model")

        if not self.model:
            return False

        # Create chat session once
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

            # Put system in history as first user message (simple)
            history = [{"role": "user", "parts": [f"SYSTEM:\n{system_prompt}"]}]
            st.session_state.gemini_chat_session = self.model.start_chat(history=history)

        self.chat_session = st.session_state.gemini_chat_session
        return True

    def chat(self, user_message: str) -> str:
        if not self.ensure_ready():
            return "⚠️ Gemini AI not available. Please add GEMINI_API_KEY in Streamlit secrets."

        try:
            resp = self.chat_session.send_message(user_message)
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                return "⚠️ Gemini returned an empty reply. Try again."
            return text
        except Exception as e:
            st.session_state.gemini_error = str(e)
            st.session_state.gemini_initialized = False
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
        "nasi lemak": {
            "calories": 600, "protein": 15, "carbs": 75, "fat": 25,
            "tips": ["Use brown rice for more fiber", "Reduce sambal to lower sodium", "Add boiled egg instead of fried chicken"],
            "alternatives": ["Nasi kerabu", "Nasi dagang", "Brown rice nasi lemak"]
        },
        "sandwich": {
            "calories": 350, "protein": 15, "carbs": 45, "fat": 12,
            "tips": ["Use whole grain bread", "Load up on vegetables", "Choose lean protein like turkey"],
            "alternatives": ["Wrap", "Salad bowl", "Open-faced sandwich"]
        },
        "pizza": {
            "calories": 285, "protein": 12, "carbs": 36, "fat": 10,
            "tips": ["Choose thin crust", "Load up on veggies", "Go easy on the cheese"],
            "alternatives": ["Cauliflower crust pizza", "Veggie pizza", "Whole wheat pizza"]
        },
        "burger": {
            "calories": 354, "protein": 20, "carbs": 35, "fat": 15,
            "tips": ["Use lettuce wrap instead of bun", "Choose lean meat", "Add lots of veggies"],
            "alternatives": ["Turkey burger", "Veggie burger", "Portobello burger"]
        },
        "pasta": {
            "calories": 220, "protein": 8, "carbs": 43, "fat": 1,
            "tips": ["Choose whole wheat pasta", "Add lean protein", "Load up on vegetables"],
            "alternatives": ["Zucchini noodles", "Whole wheat pasta", "Lentil pasta"]
        }
    }

    key = food.lower().strip()
    if key in food_data:
        return food_data[key]

    if status == "HEALTHY":
        return {
            "calories": random.randint(100, 300),
            "protein": random.randint(8, 20),
            "carbs": random.randint(10, 30),
            "fat": random.randint(3, 10),
            "tips": ["Great choice!", "Pair with protein", "Keep up the good work"],
            "alternatives": ["Similar healthy option", "Another good choice", "Variety option"]
        }
    if status == "UNHEALTHY":
        return {
            "calories": random.randint(400, 600),
            "protein": random.randint(5, 15),
            "carbs": random.randint(40, 70),
            "fat": random.randint(15, 30),
            "tips": ["Enjoy as treat", "Watch portion size", "Balance with veggies"],
            "alternatives": ["Healthier version", "Better alternative", "Light option"]
        }
    return {
        "calories": random.randint(250, 400),
        "protein": random.randint(10, 20),
        "carbs": random.randint(30, 50),
        "fat": random.randint(8, 20),
        "tips": ["Good in moderation", "Balance your meal", "Enjoy mindfully"],
        "alternatives": ["Healthier twist", "Better choice", "Alternative option"]
    }


def add_food_to_chat(food: str):
    analysis = analyze_food(food)
    tips = get_nutrition_tips(food, analysis["status"])

    st.session_state.chat_history.append({
        "type": "user", "content": food, "time": datetime.now().strftime("%H:%M")
    })
    st.session_state.chat_history.append({
        "type": "ai", "food": food, "analysis": analysis, "tips": tips, "time": datetime.now().strftime("%H:%M")
    })
    st.session_state.food_log.append({
        "food": food, "status": analysis["status"], "score": analysis["score"], "time": datetime.now()
    })


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

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🍽️ {food.upper()}")
        with col2:
            st.markdown(
                f"<span class='{analysis['badge_class']}'>{analysis['icon']} {analysis['status']}</span>",
                unsafe_allow_html=True
            )

        st.markdown(
            f'<p style="color:{analysis["color"]}; font-weight:bold; font-size:1.1rem;">{analysis["message"]}</p>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.1); padding:8px 15px; border-radius:15px; display:inline-block; margin:10px 0;'>
            📊 <b>Nutrition Score:</b> {analysis['score']}/10
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 Nutrition Facts (per serving)")

        cols = st.columns(4)
        nutrition_items = [
            ("Calories", f"{tips['calories']}", "🔥"),
            ("Protein", f"{tips['protein']}g", "💪"),
            ("Carbs", f"{tips['carbs']}g", "🌾"),
            ("Fat", f"{tips['fat']}g", "🛢️")
        ]
        for i, (label, value, icon) in enumerate(nutrition_items):
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
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.ai_chat_history.append({"sender": "user", "message": user_message, "time": timestamp})
    st.session_state.ai_chat_history.append({"sender": "ai", "message": ai_response, "time": timestamp})


# =========================
# UI Header
# =========================
st.markdown('<h1 class="main-title">🍎 BiteBot AI Nutritionist</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#00ffcc;font-size:1.2rem;margin-bottom:30px;">'
    'Instant Food Analysis • Smart Nutrition Tips • AI-Powered Chat</p>',
    unsafe_allow_html=True
)

# =========================
# Quick Stats
# =========================
if st.session_state.food_log:
    total = len(st.session_state.food_log)
    healthy = sum(1 for f in st.session_state.food_log if f["status"] == "HEALTHY")
    unhealthy = sum(1 for f in st.session_state.food_log if f["status"] == "UNHEALTHY")
    moderate = total - healthy - unhealthy

    cols = st.columns(4)
    stats = [
        ("🍽️ Total", total, "#00ccff"),
        ("✅ Healthy", healthy, "#00ff88"),
        ("⚠️ Unhealthy", unhealthy, "#ff4444"),
        ("⚖️ Moderate", moderate, "#ffcc00")
    ]
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

# -------------------------
# Tab 1: Live Chat
# -------------------------
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
    col_a, col_b = st.columns([4, 1])
    with col_a:
        user_input = st.text_input(
            "Type a food to analyze:",
            placeholder="e.g., pizza, nasi lemak, burger...",
            key="food_input",
            label_visibility="collapsed"
        )
    with col_b:
        analyze_btn = st.button("🚀 ANALYZE", use_container_width=True, type="primary")

    if analyze_btn and user_input:
        add_food_to_chat(user_input.lower())
        st.rerun()

# -------------------------
# Tab 2: Food History
# -------------------------
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
                if row["status"] == "HEALTHY":
                    st.markdown('<span class="healthy-badge" style="font-size:0.8rem; padding:5px 10px;">HEALTHY</span>', unsafe_allow_html=True)
                elif row["status"] == "UNHEALTHY":
                    st.markdown('<span class="unhealthy-badge" style="font-size:0.8rem; padding:5px 10px;">UNHEALTHY</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="moderate-badge" style="font-size:0.8rem; padding:5px 10px;">MODERATE</span>', unsafe_allow_html=True)
            with c3:
                st.progress(row["score"] / 10, text=f"{row['score']}/10")
            st.divider()

        if st.button("📥 Export History", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "bitebot_history.csv", "text/csv", key="download_csv")
    else:
        st.info("📝 No foods analyzed yet. Start by typing a food above!")

# -------------------------
# Tab 3: Dashboard
# -------------------------
with tab3:
    st.markdown("### 📈 Nutrition Dashboard")

    if st.session_state.food_log:
        df = pd.DataFrame(st.session_state.food_log)

        col1, col2 = st.columns(2)
        with col1:
            status_counts = df["status"].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, title="Food Health Distribution", hole=0.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            food_counts = df["food"].value_counts().head(10)
            fig2 = px.bar(x=food_counts.values, y=food_counts.index, orientation="h", title="Top 10 Foods Analyzed")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                               xaxis_title="Count", yaxis_title="Food")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📈 Analyze some foods to see your nutrition dashboard!")

# -------------------------
# Tab 4: AI Chat (FIXED BUGS HERE)
# -------------------------
with tab4:
    st.markdown("### 🤖 Chat with AI Nutritionist")

    # 1) Always show status + error INSIDE the tab (your code had indentation bug)
    st.caption(f"Gemini status: {st.session_state.get('gemini_initialized', False)}")
    if st.session_state.get("gemini_error"):
        st.warning(f"Gemini error: {st.session_state.gemini_error}")

    # 2) Try init (once per run) when not initialized
    if not st.session_state.get("gemini_initialized", False):
        with st.spinner("🔧 Setting up AI assistant..."):
            init_gemini()

        # show result after attempt
        if st.session_state.get("gemini_initialized", False):
            st.success("✅ Gemini AI initialized successfully!")
        else:
            st.warning("⚠️ Gemini is not initialized. Using fallback responses.")

    # Quick responses (fallback)
    fallback_responses = {
        "Give me some healthy meal ideas for weight loss": """**🥗 Healthy Meal Ideas for Weight Loss:**

**🍳 Breakfast:**
- Greek yogurt with berries & almonds
- Scrambled eggs with spinach
- Oatmeal with banana & cinnamon
- Avocado toast on whole grain

**🥗 Lunch:**
- Grilled chicken salad
- Quinoa bowl with veggies
- Turkey avocado wrap
- Lentil soup with whole grain

**🍽️ Dinner:**
- Baked salmon + broccoli
- Stir-fried tofu + veggies
- Lean beef stir-fry
- Chickpea curry + brown rice

**💡 Tips:**
- Control portions
- Include protein in every meal
- Fill half plate with vegetables
- Stay hydrated
- Choose whole foods""",

        "What are the best protein sources for muscle building?": """**💪 Best Protein Sources:**

**🥩 Animal:**
- Chicken breast
- Salmon
- Eggs
- Greek yogurt
- Lean beef

**🌱 Plant:**
- Tofu
- Lentils
- Chickpeas
- Quinoa
- Almonds

**🎯 Tips:**
- Aim 1.6–2.2g/kg/day (general guideline)
- Spread protein across meals""",

        "How can I count calories effectively?": """**🔥 Calorie Counting Tips:**
1. Track portions (scale helps)
2. Use labels + apps consistently
3. Include cooking oils + drinks
4. Review weekly trends, not daily spikes""",

        "What are common nutrition myths I should know?": """**🍎 Nutrition Myths Debunked:**
- Carbs don’t automatically make you fat
- Healthy fats are important
- You don’t need detox cleanses
- Total daily intake matters more than meal timing""",

        "How much water should I drink daily and why?": """**💧 Hydration Guide:**
- Common target: 2–3L/day (more if active/hot)
- Use urine color as a simple check (pale yellow)
- Hydration supports energy, digestion, and performance""",

        "How to create a balanced diet plan?": """**📊 Balanced Diet Plan (Plate Method):**
- ½ vegetables
- ¼ protein
- ¼ carbs (prefer whole grains)
- Add healthy fats (nuts/olive oil/avocado)
- Plan weekly + keep it sustainable"""
    }

    st.markdown("### 💡 Quick Questions")
    col_q1, col_q2 = st.columns(2)

    def ask_ai(user_msg: str):
        if st.session_state.get("gemini_initialized", False):
            resp = gemini_ai.chat(user_msg)
            if resp.startswith("⚠️"):
                resp = fallback_responses.get(user_msg, resp)
        else:
            resp = fallback_responses.get(user_msg, "⚠️ Gemini not available.")
        add_ai_chat_message(user_msg, resp)
        st.rerun()

    with col_q1:
        if st.button("🥗 Healthy Meal Ideas", use_container_width=True):
            ask_ai("Give me some healthy meal ideas for weight loss")
        if st.button("💪 Protein Sources", use_container_width=True):
            ask_ai("What are the best protein sources for muscle building?")
        if st.button("🔥 Calorie Counting", use_container_width=True):
            ask_ai("How can I count calories effectively?")

    with col_q2:
        if st.button("🍎 Food Myths", use_container_width=True):
            ask_ai("What are common nutrition myths I should know?")
        if st.button("💧 Hydration Tips", use_container_width=True):
            ask_ai("How much water should I drink daily and why?")
        if st.button("📊 Diet Planning", use_container_width=True):
            ask_ai("How to create a balanced diet plan?")

    st.divider()

    st.markdown("### 💬 Chat History")
    if not st.session_state.ai_chat_history:
        st.info("💭 Start by asking a nutrition question or using the quick questions above!")
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
    col_input1, col_input2 = st.columns([4, 1])
    with col_input1:
        ai_question = st.text_input(
            "Type your nutrition question:",
            placeholder="e.g., Good snacks for weight loss? Is intermittent fasting safe?",
            key="ai_question_input",
            label_visibility="collapsed"
        )
    with col_input2:
        ask_btn = st.button("🚀 Ask AI", use_container_width=True, type="primary")

    if ask_btn and ai_question:
        if st.session_state.get("gemini_initialized", False):
            response = gemini_ai.chat(ai_question)
            if response.startswith("⚠️"):
                response = (
                    f"**Nutrition advice about '{ai_question}':**\n\n"
                    "For personalized advice:\n"
                    "1. Consult a registered dietitian\n"
                    "2. Use evidence-based sources\n"
                    "3. Consider your health conditions\n"
                    "4. Make sustainable changes"
                )
        else:
            response = (
                f"**Nutrition advice about '{ai_question}':**\n\n"
                "Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets.\n\n"
                "General tips:\n"
                "1. Whole foods\n2. Protein + fiber\n3. Portion control\n4. Hydration"
            )

        add_ai_chat_message(ai_question, response)
        st.rerun()

    # Clear AI Chat button (also clears gemini chat session)
    if st.button("🗑️ Clear AI Chat", use_container_width=True, type="secondary"):
        st.session_state.ai_chat_history = []
        st.session_state.gemini_chat_session = None
        st.rerun()

# =========================
# Clear All
# =========================
if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
    st.session_state.chat_history = []
    st.session_state.food_log = []
    st.session_state.ai_chat_history = []
    st.session_state.gemini_chat_session = None
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;padding:20px;">
    <p>🍎 <b>BiteBot AI Nutritionist</b> • Smart Food Analysis • Real-time Chat • Powered by AI</p>
    <p style="font-size:0.9rem;">Instant food analysis + AI-powered nutrition advice</p>
</div>
""", unsafe_allow_html=True)
