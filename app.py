import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import os
from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


    

def safe_to_float(x):
    """Try several ways to coerce x to a Python float. Return np.nan on failure."""
    try:
        arr = np.array(x)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            try:
                return float(pd.to_numeric(x, errors="coerce"))
            except Exception:
                return np.nan

# --- cache translator ---
@st.cache_resource
def get_translator(target_lang="en"):
    return GoogleTranslator(source="auto", target=target_lang)

def translate_text(text, target_lang="en"):
    """Translate text using cached GoogleTranslator"""
    try:
        if not text:
            return ""
        translator = get_translator(target_lang)
        return translator.translate(text)
    except Exception:
        return text

# -------------------------
# Page + style
# -------------------------
st.set_page_config(page_title="Farm Sevak Agriculture Predictor", page_icon="🌾", layout="centered")

# Sidebar language selector
st.sidebar.header("🌐 Language")
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Odia": "or"
}
selected_lang = st.sidebar.selectbox("Choose Language", list(lang_map.keys()))
TARGET_LANG = lang_map[selected_lang]


# -------------------------
# Weather API integration
# -------------------------

def get_weather(city_name):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rainfall": data["rain"]["1h"] if "rain" in data else 0
            }
        else:
            return None
    except Exception:
        return None

# Sidebar Weather Fetch
st.sidebar.markdown("### 🌦️ Weather Data")
city_input = st.sidebar.text_input("Enter City Name")

if st.sidebar.button("Fetch Weather"):
    weather = get_weather(city_input)
    if weather:
        st.sidebar.success(f"🌡️ Temp: {weather['temperature']} °C, 💧 Humidity: {weather['humidity']} %, 🌧️ Rainfall: {weather['rainfall']} mm")
        # Autofill values in the main inputs
        temperature = weather["temperature"]
        humidity = weather["humidity"]
        annual_rain = weather["rainfall"]
    else:
        st.sidebar.error("⚠️ Could not fetch weather. Check city name or API key.")


# -------------------------
# Load model from Dropbox
# -------------------------
MODEL_URL = "https://www.dropbox.com/scl/fi/rqmel07pl1hswv4eutgly/Yield_prediction.pk1?rlkey=i2fk2n9ypvgt6i8jyfidqnlr4&st=ejb97kr9&dl=1"
MODEL_PATH = "Yield_Prediction.pk1"

# --- cache model loading ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            st.error("Failed to download model from Dropbox.")
            st.exception(e)
            st.stop()
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("Error loading model file.")
        st.exception(e)
        st.stop()

model = load_model()

# -------------------------
# Header + Inputs
# -------------------------
st.title(translate_text("🌾 Farm Sevak Agriculture Predictor", TARGET_LANG))
st.markdown(f'<div class="subtitle">{translate_text("An interactive web app built with Machine Learning to predict crop yield, fertilizer, and pesticide requirements based on farm and soil parameters.", TARGET_LANG)}</div>', unsafe_allow_html=True)

state_options = ["Andhra Pradesh","Bihar","Gujarat","Karnataka","Maharashtra","Punjab","Rajasthan","Tamil Nadu","Uttar Pradesh","West Bengal","Other"]
crop_options = ["Rice","Wheat","Maize","Cotton","Sugarcane","Soybean","Pulses","Other"]

col1, col2 = st.columns([1, 1])
with col1:
    crop_sel = st.selectbox("🌾 " + translate_text("Crop", TARGET_LANG), crop_options)
    crop = st.text_input(translate_text("Type Crop (manual)", TARGET_LANG)) if crop_sel == "Other" else crop_sel

    season = st.selectbox("☀️ " + translate_text("Season", TARGET_LANG), ["Kharif","Rabi","Summer","Whole Year"])
    state_sel = st.selectbox("🏞️ " + translate_text("State", TARGET_LANG), state_options)
    state = st.text_input(translate_text("Type State (manual)", TARGET_LANG)) if state_sel == "Other" else state_sel

with col2:
    area = st.number_input("📏 " + translate_text("Area (hectares)", TARGET_LANG), min_value=0.0, step=0.1, value=1.0, format="%.3f")
    annual_rain = st.number_input("🌧️ " + translate_text("Annual Rainfall (mm)", TARGET_LANG), min_value=0.0, step=0.1, value=500.0)
    temperature = st.number_input("🌡️ " + translate_text("Temperature (°C)", TARGET_LANG), value=25.0, step=0.1)
    humidity = st.number_input("💧 " + translate_text("Humidity (%)", TARGET_LANG), min_value=0.0, max_value=100.0, value=60.0, step=0.1)

col3, col4 = st.columns([1,1])
with col3:
    ph = st.number_input("🧪 " + translate_text("pH", TARGET_LANG), min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    n_percent = st.number_input("🟢 " + translate_text("N_percent (%)", TARGET_LANG), min_value=0.0, step=0.1, value=0.5)
with col4:
    p_percent = st.number_input("🔴 " + translate_text("P_percent (%)", TARGET_LANG), min_value=0.0, step=0.1, value=0.2)
    k_percent = st.number_input("🟡 " + translate_text("K_percent (%)", TARGET_LANG), min_value=0.0, step=0.1, value=0.3)

soil_type = st.selectbox("🌍 " + translate_text("Soil Type", TARGET_LANG), ["Alluvial","Black","Red","Laterite","Sandy","Clay"])
st.markdown(f'<div class="small_note">{translate_text("Choose from dropdown or type custom Crop/State above.", TARGET_LANG)}</div>', unsafe_allow_html=True)

# -------------------------
# Predict button
# -------------------------
if st.button("🔎 " + translate_text("Predict", TARGET_LANG)):
    input_df = pd.DataFrame([{
        "Crop": crop,
        "Season": season,
        "State": state,
        "Area": area,
        "Annual_Rainfall": annual_rain,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "N_percent": n_percent,
        "P_percent": p_percent,
        "K_percent": k_percent,
        "Soil_Type": soil_type
    }])

    try:
        raw_pred = model.predict(input_df)
    except Exception as e:
        st.error(translate_text("Model prediction failed. Make sure your pipeline accepts a DataFrame with these column names.", TARGET_LANG))
        st.exception(e)
        st.stop()

    if isinstance(raw_pred, pd.DataFrame):
        pred_row = raw_pred.iloc[0].values
    else:
        pred_row = np.array(raw_pred).ravel()

    if pred_row.size != 6:
        st.error(translate_text(f"Model returned {pred_row.size} outputs but 6 were expected.", TARGET_LANG))
        st.stop()

    yield_per_ha = safe_to_float(pred_row[0])
    total_yield = safe_to_float(pred_row[1])
    fert_per_ha = safe_to_float(pred_row[2])
    pest_per_ha = safe_to_float(pred_row[3])
    fertilizer = safe_to_float(pred_row[4])
    pesticide = safe_to_float(pred_row[5])

    try:
        area_f = float(area)
    except Exception:
        st.error(translate_text("Area must be numeric.", TARGET_LANG))
        st.stop()

    total_yield = yield_per_ha * area_f
    fertilizer = fert_per_ha * area_f
    pesticide = pest_per_ha * area_f

    st.markdown("---")
    st.write(f"- {translate_text('Yield per ha', TARGET_LANG)} : `{yield_per_ha:.3f}` t/ha")
    st.write(f"- {translate_text('Total Yield', TARGET_LANG)} : `{total_yield:.3f}` t ")
    st.write(f"- {translate_text('Fertilizer per ha', TARGET_LANG)} : `{fert_per_ha:.3f}` kg/ha")
    st.write(f"- {translate_text('Total Fertilizer', TARGET_LANG)} : `{fertilizer:.3f}` kg ")
    st.write(f"- {translate_text('Pesticide per ha', TARGET_LANG)} : `{pest_per_ha:.3f}` kg/ha")
    st.write(f"- {translate_text('Total Pesticide', TARGET_LANG)} : `{pesticide:.3f}` kg ")

    comp_df = pd.DataFrame({
        "Metric": [translate_text("Total Yield (t)", TARGET_LANG), translate_text("Fertilizer (kg)", TARGET_LANG), translate_text("Pesticide (kg)", TARGET_LANG)],
        "Calculated": [total_yield, fertilizer, pesticide]
    })
    comp_melt = comp_df.melt(id_vars="Metric", var_name="Source", value_name="Value")

    fig = px.bar(
        comp_melt,
        x="Metric",
        y="Value",
        color="Metric",
        text=comp_melt["Value"].round(2),
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Agsunset,
    )
    fig.update_traces(
        textposition="outside",
        marker=dict(line=dict(width=1, color="white"), opacity=0.85)
    )
    fig.update_layout(
        title=translate_text("📊 Yield, Fertilizer & Pesticide Comparison", TARGET_LANG),
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=translate_text("Metrics", TARGET_LANG), showgrid=False),
        yaxis=dict(title=translate_text("Values", TARGET_LANG), showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        margin=dict(t=60, b=40, l=40, r=40),
        bargap=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ " + translate_text("Done — calculations shown above.", TARGET_LANG))

    # ---------------- LLM Setup ----------------
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
    chat_model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    
    prompt = PromptTemplate(
        input_variables=[
            "crop", "season", "state", "area", "annual_rain", "temperature", "humidity",
            "soil_type", "ph", "n_percent", "p_percent", "k_percent",
            "total_yield", "fertilizer", "pesticide"
        ],
        template="""
You are an agricultural expert. Based on the details below, give a farmer-friendly recommendation.

Details:
- Crop: {crop}, Season: {season}, State: {state}, Area: {area} ha
- Rainfall: {annual_rain} mm, Temp: {temperature} °C, Humidity: {humidity} %
- Soil: {soil_type}, pH: {ph}, N: {n_percent}, P: {p_percent}, K: {k_percent}
- Predicted Yield: {total_yield} tons, Fertilizer: {fertilizer} kg, Pesticide: {pesticide} L

👉 Task: Give **max 8 short bullet points** with advice on crop choice, fertilizer, pesticide, irrigation, soil care, risks, and profit.
"""
    )

    chain = prompt | chat_model | parser

    # ---------------- LLM Recommendation ----------------
    st.markdown("---")
    st.subheader(translate_text("🌱 Expert Recommendation", TARGET_LANG))

    prompt_inputs = {
        "crop": crop,
        "season": season,
        "state": state,
        "area": area,
        "annual_rain": annual_rain,
        "temperature": temperature,
        "humidity": humidity,
        "soil_type": soil_type,
        "ph": ph,
        "n_percent": n_percent,
        "p_percent": p_percent,
        "k_percent": k_percent,
        "total_yield": total_yield,
        "fertilizer": fertilizer,
        "pesticide": pesticide,
    }

    try:
        recommendation = chain.invoke(prompt_inputs)
        recommendation_translated = translate_text(recommendation, TARGET_LANG)
        st.markdown(recommendation_translated)
        # ---------------- PDF Download ----------------
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        # Create PDF in memory
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "🌱 Expert Recommendation Report")

        # Add recommendation text (wrap lines)
        c.setFont("Helvetica", 11)
        y = height - 80
        for line in recommendation_translated.split("\n"):
            if y < 50:  # new page if space ends
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(50, y, line.strip())
            y -= 20

        c.save()
        buffer.seek(0)

        # Streamlit download button
        st.download_button(
            label="📥 Download Recommendation as PDF",
            data=buffer,
            file_name="recommendation_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(translate_text("LLM recommendation failed.", TARGET_LANG))
        st.exception(e)



    
