# 🍎 BiteBot AI Nutritionist

BiteBot is a deployed, AI-powered food health recommendation system that combines a **machine learning classification model (Logistic Regression)** with **generative AI (Google Gemini)** to provide food health predictions, nutrition insights, and conversational explanations through a web-based interface.

---

## 📌 Project Overview

Many individuals struggle to quickly determine whether their meals are healthy, especially when consuming food from mixed sources such as home-cooked meals, hawker stalls, restaurants, and packaged foods. BiteBot addresses this issue by allowing users to input food items and receive an immediate health assessment along with clear explanations and improvement suggestions.

The system integrates a **traditional machine learning model** for food health prediction with a **large language model (LLM)** to improve interpretability and user experience.

---

## 🧠 Machine Learning Component

This project implements a **supervised machine learning classification model using Logistic Regression**.

### Model Details:
- **Algorithm used:** Logistic Regression
- **Learning type:** Supervised learning
- **Task:** Binary / categorical classification
- **Input features:** Nutritional attributes such as calories, fat, sugar, carbohydrates, and protein
- **Output:** Food health classification (Healthy / Moderate / Unhealthy) and nutrition score

### Machine Learning Process:
1. Nutrition-related data is analysed and preprocessed.
2. A **Logistic Regression model** is applied to learn the relationship between nutritional features and food health outcomes.
3. The trained model produces consistent and interpretable predictions.
4. Model outputs are used as the decision foundation of the system.

Logistic Regression was selected due to its **simplicity, interpretability, and suitability for classification tasks** in nutrition-based applications.

---

## 🤖 Generative AI (LLM) Component

To enhance user understanding, BiteBot integrates **Google Gemini**, a large language model, to:

- Explain the Logistic Regression predictions in natural language
- Provide personalised nutrition advice
- Answer general food and health-related questions
- Enable conversational interaction via a chatbot interface

This results in a **hybrid ML + LLM architecture**, where traditional machine learning predictions are explained and enhanced using generative AI.

---

## 🌐 Deployment

BiteBot is **fully deployed as a web application** using **Streamlit Cloud**.

### Deployment Highlights:
- Publicly accessible web application
- Connected to a GitHub repository for version control
- Secure API key handling via Streamlit Secrets
- No local installation required for end users

This deployment demonstrates a complete **end-to-end AI system**, from model usage to production deployment.

---

## 📊 Application Features

- Live food input and instant health classification
- Logistic Regression–based prediction logic
- Nutrition score and food category indicators
- AI-powered conversational chat (Gemini)
- Food history tracking with CSV export
- Interactive dashboard with visual analytics

---

## 🛠️ Tools and Technologies

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning Model:** Logistic Regression  
- **Libraries:** pandas, numpy, scikit-learn  
- **Visualization:** Plotly  
- **LLM API:** Google Gemini  
- **Deployment Platform:** Streamlit Cloud  
- **Version Control:** GitHub  

---

## 🔑 Environment Configuration

The Gemini API key is securely stored using environment variables.

