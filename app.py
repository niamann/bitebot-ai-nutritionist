import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import random
import time
import os
from dotenv import load_dotenv
import json
from typing import Dict, List
import re

# Page config
st.set_page_config(
    page_title="🍎 BiteBot AI Nutritionist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS - Only minimal styling
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
    
    .food-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .nutrition-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    
    .nutrition-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    
    .nutrition-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ffcc;
        margin: 5px 0;
    }
    
    .nutrition-label {
        font-size: 0.8rem;
        color: #aaa;
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

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'food_log' not in st.session_state:
    st.session_state.food_log = []
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []
if 'gemini_initialized' not in st.session_state:
    st.session_state.gemini_initialized = False

# FOOD DATABASE
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

def init_gemini():
    """Initialize Gemini client using GEMINI_API_KEY from Streamlit secrets or environment."""
    try:
        load_dotenv()

        api_key = None

        # 1) Streamlit Cloud Secrets
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None

        # 2) Local env / .env
        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.error("GEMINI_API_KEY not found. Add it to Streamlit Secrets or set it in your environment.")
            st.session_state.gemini_initialized = False
            return None

        client = genai.Client(api_key=api_key)

        st.session_state.gemini_initialized = True
        st.session_state.gemini_client = client
        return client

    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {str(e)}")
        st.session_state.gemini_initialized = False
        return None


# Simple Gemini AI Chat Class
class GeminiNutritionAI:
    def __init__(self):
        self.client = None

    def start_chat(self):
        """Prepare Gemini client + initialize chat history."""
        if not st.session_state.get("gemini_initialized", False):
            self.client = init_gemini()
        else:
            self.client = st.session_state.get("gemini_client")

        if not self.client:
            return False

        if "gemini_messages" not in st.session_state:
            # We'll store messages as a simple list of dicts
            st.session_state.gemini_messages = []

        # Put your system prompt once (first time only)
        if not st.session_state.gemini_messages:
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
            st.session_state.gemini_messages.append({"role": "system", "text": system_prompt})

        return True

    def chat(self, user_message: str) -> str:
        """Send message to Gemini and get response text."""
        if not self.client:
            if not self.start_chat():
                return "⚠️ Gemini AI is not available at the moment. Using fallback responses."

        try:
            # Add user message
            st.session_state.gemini_messages.append({"role": "user", "text": user_message})

            # Build a single text prompt from history (simple + reliable)
            prompt_lines = []
            for m in st.session_state.gemini_messages:
                if m["role"] == "system":
                    prompt_lines.append(f"SYSTEM:\n{m['text']}\n")
                elif m["role"] == "user":
                    prompt_lines.append(f"USER: {m['text']}\n")
                else:
                    prompt_lines.append(f"ASSISTANT: {m['text']}\n")

            combined_prompt = "\n".join(prompt_lines) + "\nASSISTANT:"

            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=combined_prompt,
            )

            ai_text = (resp.text or "").strip()
            if not ai_text:
                ai_text = "⚠️ I didn't get a response. Using fallback information."

            # Save assistant reply to history
            st.session_state.gemini_messages.append({"role": "assistant", "text": ai_text})

            return ai_text

        except Exception as e:
            # If Gemini fails, use fallback based on the question
            error_msg = str(e)
            # Don't show technical error to user, just use fallback
            return "⚠️ Gemini AI is currently busy. Using our pre-built nutrition information instead."

# Initialize Gemini AI
gemini_ai = GeminiNutritionAI()

def analyze_food(food_name):
    """Analyze food and return health status"""
    food_lower = food_name.lower()
    
    # Check each category
    for category in ["healthy", "unhealthy", "moderate"]:
        for food, score in FOOD_DATABASE[category].items():
            if food in food_lower:
                if category == "healthy":
                    return {
                        "status": "HEALTHY",
                        "score": score,
                        "badge_class": "healthy-badge",
                        "icon": "✅",
                        "message": "🥗 EXCELLENT! This is super nutritious!",
                        "color": "#00ff88"
                    }
                elif category == "unhealthy":
                    return {
                        "status": "UNHEALTHY",
                        "score": score,
                        "badge_class": "unhealthy-badge",
                        "icon": "⚠️",
                        "message": "🔴 Enjoy occasionally in small portions.",
                        "color": "#ff4444"
                    }
                else:
                    return {
                        "status": "MODERATE",
                        "score": score,
                        "badge_class": "moderate-badge",
                        "icon": "⚖️",
                        "message": "🟡 Good in moderation with balanced diet.",
                        "color": "#ffcc00"
                    }
    
    # Default to moderate
    return {
        "status": "MODERATE",
        "score": random.randint(5, 7),
        "badge_class": "moderate-badge",
        "icon": "⚖️",
        "message": "🟡 Enjoy as part of balanced meals.",
        "color": "#ffcc00"
    }

def get_nutrition_tips(food, status):
    """Get nutrition tips for the food"""
    
    # Specific food data
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
    
    food_lower = food.lower()
    if food_lower in food_data:
        data = food_data[food_lower]
    else:
        # General nutrition estimates
        if status == "HEALTHY":
            data = {
                "calories": random.randint(100, 300),
                "protein": random.randint(8, 20),
                "carbs": random.randint(10, 30),
                "fat": random.randint(3, 10),
                "tips": ["Great choice!", "Pair with protein", "Keep up the good work"],
                "alternatives": ["Similar healthy option", "Another good choice", "Variety option"]
            }
        elif status == "UNHEALTHY":
            data = {
                "calories": random.randint(400, 600),
                "protein": random.randint(5, 15),
                "carbs": random.randint(40, 70),
                "fat": random.randint(15, 30),
                "tips": ["Enjoy as treat", "Watch portion size", "Balance with veggies"],
                "alternatives": ["Healthier version", "Better alternative", "Light option"]
            }
        else:  # MODERATE
            data = {
                "calories": random.randint(250, 400),
                "protein": random.randint(10, 20),
                "carbs": random.randint(30, 50),
                "fat": random.randint(8, 20),
                "tips": ["Good in moderation", "Balance your meal", "Enjoy mindfully"],
                "alternatives": ["Healthier twist", "Better choice", "Alternative option"]
            }
    
    return data

def add_food_to_chat(food):
    """Add food analysis to chat"""
    # Analyze food
    analysis = analyze_food(food)
    tips = get_nutrition_tips(food, analysis["status"])
    
    # Add to chat history
    st.session_state.chat_history.append({
        "type": "user",
        "content": food,
        "time": datetime.now().strftime("%H:%M")
    })
    
    st.session_state.chat_history.append({
        "type": "ai",
        "food": food,
        "analysis": analysis,
        "tips": tips,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Add to food log
    st.session_state.food_log.append({
        "food": food,
        "status": analysis["status"],
        "score": analysis["score"],
        "time": datetime.now()
    })

def display_ai_response(food, analysis, tips):
    """Display AI response using Streamlit components"""
    
    # AI message container
    with st.container():
        st.markdown(f"""
        <div class='ai-message'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                <div style='font-weight:bold; color:#0088ff;'>🤖 BiteBot AI</div>
                <div style='font-size:0.8rem; color:#aaa;'>{datetime.now().strftime("%H:%M")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Food title and status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🍽️ {food.upper()}")
        with col2:
            badge_html = f"""
            <span class='{analysis["badge_class"]}'>
                {analysis["icon"]} {analysis["status"]}
            </span>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
        
        # Message
        st.markdown(f'<p style="color:{analysis["color"]}; font-weight:bold; font-size:1.1rem;">{analysis["message"]}</p>', unsafe_allow_html=True)
        
        # Nutrition Score
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.1); padding:8px 15px; border-radius:15px; display:inline-block; margin:10px 0;'>
            📊 <b>Nutrition Score:</b> {analysis['score']}/10
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Nutrition Facts
        st.markdown("#### 📊 Nutrition Facts (per serving)")
        
        # Create nutrition grid using st.columns
        cols = st.columns(4)
        nutrition_items = [
            ("Calories", f"{tips['calories']}", "🔥"),
            ("Protein", f"{tips['protein']}g", "💪"),
            ("Carbs", f"{tips['carbs']}g", "🌾"),
            ("Fat", f"{tips['fat']}g", "🛢️")
        ]
        
        for idx, (label, value, icon) in enumerate(nutrition_items):
            with cols[idx]:
                st.markdown(f"""
                <div style='text-align:center; background:rgba(255,255,255,0.05); padding:15px; border-radius:8px;'>
                    <div style='font-size:1.5rem;'>{icon}</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#00ffcc; margin:5px 0;'>{value}</div>
                    <div style='font-size:0.9rem; color:#aaa;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Smart Tips
        st.markdown("#### 💡 Smart Eating Tips")
        for tip in tips['tips']:
            st.markdown(f"• {tip}")
        
        st.divider()
        
        # Healthier Alternatives
        st.markdown("#### 🔄 Healthier Alternatives")
        alt_cols = st.columns(3)
        for idx, alt in enumerate(tips['alternatives']):
            with alt_cols[idx]:
                st.info(alt)
        
        st.divider()
        
        # Pro Advice
        st.success("🎯 **Pro Advice:** Eat mindfully • Stay hydrated • Enjoy your food • Listen to your body")

def add_ai_chat_message(user_message, ai_response):
    """Add AI chat message to history"""
    timestamp = datetime.now().strftime("%H:%M")
    
    # Add user message
    st.session_state.ai_chat_history.append({
        "sender": "user",
        "message": user_message,
        "time": timestamp
    })
    
    # Add AI response
    st.session_state.ai_chat_history.append({
        "sender": "ai",
        "message": ai_response,
        "time": timestamp
    })

# Main App
st.markdown('<h1 class="main-title">🍎 BiteBot AI Nutritionist</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#00ffcc;font-size:1.2rem;margin-bottom:30px;">Instant Food Analysis • Smart Nutrition Tips • AI-Powered Chat</p>', unsafe_allow_html=True)

# Quick Stats
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

# Tabs - Added AI Chat tab
tab1, tab2, tab3, tab4 = st.tabs(["💬 Live Chat", "📊 Food History", "📈 Dashboard", "🤖 AI Chat"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Live Chat")
        
        # Display chat history
        if not st.session_state.chat_history:
            st.info("💬 Start by typing a food or clicking a quick food button!")
        else:
            for msg in st.session_state.chat_history:
                if msg["type"] == "user":
                    # Display user message
                    st.markdown(f"""
                    <div class='user-message'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;'>
                            <div style='font-weight:bold; color:#00ffcc;'>👤 YOU</div>
                            <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time', '')}</div>
                        </div>
                        <div style='font-size:1.1rem;'>🍽️ <b>{msg['content'].upper()}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif msg["type"] == "ai":
                    # Display AI response using the new function
                    display_ai_response(msg["food"], msg["analysis"], msg["tips"])
    
    with col2:
        st.markdown("### 🍎 Quick Foods")
        
        # Quick food buttons
        quick_foods = [
            ("🍕", "pizza"),
            ("🥗", "salad"),
            ("🍔", "burger"),
            ("🍣", "sushi"),
            ("🍝", "pasta"),
            ("🍫", "chocolate"),
            ("🍦", "ice cream"),
            ("🍎", "apple"),
            ("🥤", "soda"),
            ("🍗", "chicken"),
            ("🐟", "fish"),
            ("🥑", "avocado"),
            ("🍚", "nasi lemak"),
            ("🍜", "ramen"),
            ("🥪", "sandwich")
        ]
        
        # Create buttons in a single column
        for emoji, food in quick_foods:
            if st.button(f"{emoji} {food.title()}", key=f"quick_{food}", use_container_width=True):
                add_food_to_chat(food)
                st.rerun()
    
    # Input Section
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

with tab2:
    st.markdown("### 📊 Food History")
    
    if st.session_state.food_log:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.food_log)
        df = df.sort_values("time", ascending=False)
        
        # Display each entry
        for idx, row in df.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{row['food'].title()}**")
                st.caption(f"{row['time'].strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                if row["status"] == "HEALTHY":
                    st.markdown('<span class="healthy-badge" style="font-size:0.8rem; padding:5px 10px;">HEALTHY</span>', unsafe_allow_html=True)
                elif row["status"] == "UNHEALTHY":
                    st.markdown('<span class="unhealthy-badge" style="font-size:0.8rem; padding:5px 10px;">UNHEALTHY</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="moderate-badge" style="font-size:0.8rem; padding:5px 10px;">MODERATE</span>', unsafe_allow_html=True)
            
            with col3:
                st.progress(row['score'] / 10, text=f"{row['score']}/10")
            
            st.divider()
        
        # Export button
        if st.button("📥 Export History", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "bitebot_history.csv",
                "text/csv",
                key="download_csv"
            )
    else:
        st.info("📝 No foods analyzed yet. Start by typing a food above!")

with tab3:
    st.markdown("### 📈 Nutrition Dashboard")
    
    if st.session_state.food_log and len(st.session_state.food_log) > 0:
        df = pd.DataFrame(st.session_state.food_log)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            status_counts = df["status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Food Health Distribution",
                color=status_counts.index,
                color_discrete_map={
                    "HEALTHY": "#00ff88",
                    "UNHEALTHY": "#ff4444",
                    "MODERATE": "#ffcc00"
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            food_counts = df["food"].value_counts().head(10)
            fig2 = px.bar(
                x=food_counts.values,
                y=food_counts.index,
                orientation='h',
                title="Top 10 Foods Analyzed",
                color=food_counts.values,
                color_continuous_scale="Viridis"
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Count",
                yaxis_title="Food"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📈 Analyze some foods to see your nutrition dashboard!")

with tab4:
    st.markdown("### 🤖 Chat with AI Nutritionist")
    
    # Try to initialize Gemini automatically
    if not st.session_state.gemini_initialized:
        with st.spinner("🔧 Setting up AI assistant..."):
            try:
                init_gemini()
                if st.session_state.gemini_initialized:
                    st.success("✅ Gemini AI initialized successfully!")
                else:
                    st.warning("⚠️ Could not initialize Gemini AI. Using fallback responses.")
            except:
                st.warning("⚠️ Could not initialize Gemini AI. Using fallback responses.")
    
    # Display AI Chat Interface
    st.markdown("### 💡 Quick Questions")
    col_q1, col_q2 = st.columns(2)
    
    # Fallback responses if Gemini fails - UPDATED TO BE MORE ROBUST
    fallback_responses = {
        "Give me some healthy meal ideas for weight loss": """
**🥗 Healthy Meal Ideas for Weight Loss:**

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
        
        "What are the best protein sources for muscle building?": """
**💪 Best Protein Sources:**

**🥩 Animal:**
- Chicken breast (31g/100g)
- Salmon (22g/100g)
- Eggs (6g/egg)
- Greek yogurt (10g/100g)
- Lean beef (26g/100g)

**🌱 Plant:**
- Tofu (8g/100g)
- Lentils (9g/100g)
- Chickpeas (9g/100g)
- Quinoa (4g/100g)
- Almonds (21g/100g)

**⏰ Timing:**
- Eat protein after workout
- Distribute throughout day
- Aim 1.6-2.2g/kg body weight
- Combine plant proteins""",
        
        "How can I count calories effectively?": """
**🔥 Calorie Counting Tips:**

**📱 Tools:**
- MyFitnessPal app
- Kitchen food scale
- Measuring cups/spoons
- Nutrition labels

**📊 How to Track:**
1. Calculate your TDEE
2. Set 500-750 calorie deficit
3. Track everything you eat
4. Be consistent
5. Adjust weekly

**🎯 Strategies:**
- Meal prep in advance
- Read labels carefully
- Account for cooking oils
- Don't forget drinks
- Be patient""",
        
        "What are common nutrition myths I should know?": """
**🍎 Nutrition Myths Debunked:**

**❌ Myth 1:** Carbs make you fat
**✅ Truth:** Excess calories make you fat

**❌ Myth 2:** Eating fat makes you fat
**✅ Truth:** Healthy fats are essential

**❌ Myth 3:** Need detox juice cleanses
**✅ Truth:** Liver/kidneys detox naturally

**❌ Myth 4:** Eating late causes weight gain
**✅ Truth:** Total daily calories matter

**❌ Myth 5:** All calories are equal
**✅ Truth:** Nutrient quality matters""",
        
        "How much water should I drink daily and why?": """
**💧 Water Intake Guide:**

**💧 How Much:**
- 2-3 liters (8-12 cups) daily
- More if exercising/hot climate
- Listen to thirst signals
- Monitor urine color (pale yellow)

**🚰 Benefits:**
- Improves energy
- Enhances brain function
- Supports digestion
- Aids weight management
- Improves skin

**🥤 Drink More:**
- Carry water bottle
- Set reminders
- Add lemon/cucumber
- Eat water-rich foods
- Track intake""",
        
        "How to create a balanced diet plan?": """
**📊 Balanced Diet Plan:**

**🍎 Plate Method:**
- ½ Plate: Vegetables
- ¼ Plate: Protein
- ¼ Plate: Complex carbs
- Small side: Healthy fats

**📅 Weekly Planning:**
1. Plan meals weekly
2. Make shopping list
3. Prep ingredients
4. Cook in batches
5. Include variety

**🎯 Principles:**
- Eat whole foods
- Include all food groups
- Practice portion control
- Stay consistent
- Allow occasional treats"""
    }
    
    with col_q1:
        if st.button("🥗 Healthy Meal Ideas", use_container_width=True):
            user_msg = "Give me some healthy meal ideas for weight loss"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    # Check if response is error
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here's some healthy meal ideas...")
                else:
                    response = fallback_responses.get(user_msg, "Here's some healthy meal ideas...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
        
        if st.button("💪 Protein Sources", use_container_width=True):
            user_msg = "What are the best protein sources for muscle building?"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here are protein sources...")
                else:
                    response = fallback_responses.get(user_msg, "Here are protein sources...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
        
        if st.button("🔥 Calorie Counting", use_container_width=True):
            user_msg = "How can I count calories effectively?"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here's how to count calories...")
                else:
                    response = fallback_responses.get(user_msg, "Here's how to count calories...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
    
    with col_q2:
        if st.button("🍎 Food Myths", use_container_width=True):
            user_msg = "What are common nutrition myths I should know?"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here are nutrition myths...")
                else:
                    response = fallback_responses.get(user_msg, "Here are nutrition myths...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
        
        if st.button("💧 Hydration Tips", use_container_width=True):
            user_msg = "How much water should I drink daily and why?"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here's about hydration...")
                else:
                    response = fallback_responses.get(user_msg, "Here's about hydration...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
        
        if st.button("📊 Diet Planning", use_container_width=True):
            user_msg = "How to create a balanced diet plan?"
            with st.spinner("Thinking..."):
                if st.session_state.gemini_initialized:
                    response = gemini_ai.chat(user_msg)
                    if "⚠️" in response:
                        response = fallback_responses.get(user_msg, "Here's about diet planning...")
                else:
                    response = fallback_responses.get(user_msg, "Here's about diet planning...")
                add_ai_chat_message(user_msg, response)
            st.rerun()
    
    st.divider()
    
    # Display AI Chat History
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
                        <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time', '')}</div>
                    </div>
                    <div style='font-size:1.1rem;'>{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='gemini-message'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                        <div style='font-weight:bold;' class='gemini-title'>🤖 BiteBot AI</div>
                        <div style='font-size:0.8rem; color:#aaa;'>{msg.get('time', '')}</div>
                    </div>
                    <div style='font-size:1.1rem;'>{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Input for custom questions
    st.markdown("### 💭 Ask Your Question")
    col_input1, col_input2 = st.columns([4, 1])
    
    with col_input1:
        ai_question = st.text_input(
            "Type your nutrition question:",
            placeholder="e.g., How to lose weight healthily? What are good snacks? Is intermittent fasting safe?",
            key="ai_question_input",
            label_visibility="collapsed"
        )
    
    with col_input2:
        ask_btn = st.button("🚀 Ask AI", use_container_width=True, type="primary")
    
    if ask_btn and ai_question:
        with st.spinner("🤖 AI is thinking..."):
            if st.session_state.gemini_initialized:
                response = gemini_ai.chat(ai_question)
                # If Gemini returns error, use generic fallback
                if "⚠️" in response:
                    response = f"**Here's nutrition advice about '{ai_question}':**\n\nFor personalized advice, I recommend:\n1. Consulting with a registered dietitian\n2. Looking at evidence-based nutrition sources\n3. Considering your individual health needs\n4. Making sustainable lifestyle changes"
            else:
                # Simple fallback for custom questions
                response = f"**Nutrition advice about '{ai_question}':**\n\nFor detailed answers, ensure Gemini AI is configured.\n\nI recommend:\n1. Consulting with a registered dietitian\n2. Looking at evidence-based nutrition sources\n3. Considering your individual health needs\n4. Making sustainable lifestyle changes"
            add_ai_chat_message(ai_question, response)
        st.rerun()
    
    # Clear AI Chat button
    if st.button("🗑️ Clear AI Chat", use_container_width=True, type="secondary"):
        st.session_state.ai_chat_history = []
        if "gemini_messages" in st.session_state:
            st.session_state.gemini_messages = []
        st.rerun()

# Clear Chat Button
if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
    st.session_state.chat_history = []
    st.session_state.food_log = []
    st.session_state.ai_chat_history = []
    if "gemini_messages" in st.session_state:
        st.session_state.gemini_messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;padding:20px;">
    <p>🍎 <b>BiteBot AI Nutritionist</b> • Smart Food Analysis • Real-time Chat • Powered by AI</p>
    <p style="font-size:0.9rem;">Instant food analysis + AI-powered nutrition advice</p>
</div>
""", unsafe_allow_html=True)
