import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ast
import google.generativeai as genai
# Setup

st.set_page_config(page_title="TSLA Candlestick Dashboard", layout="wide")
st.title("📈 TSLA Dashboard + Chatbot")
# ---- GEMINI API SETUP ----
api = "AIzaSyBSTRg5GFj4UGXI88UVKngIyhOyyv_ox6Q"  # Replace with your working API key
genai.configure(api_key=api)
model = genai.GenerativeModel('models/gemini-1.5-flash')
# ---- LOAD & CLEAN DATA ----
df = pd.read_csv("tsla_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fix literal eval for Support/Resistance
df['Support'] = df['Support'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['Resistance'] = df['Resistance'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
# Load CSV
# @st.cache_data
def load_data():
    df = pd.read_csv("tsla_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert stringified lists to actual lists
    df['Support'] = df['Support'].apply(eval)
    df['Resistance'] = df['Resistance'].apply(eval)
    return df

df = load_data()

# Generate global context for Gemini
# @st.cache_data
def generate_gemini_context(df):
    highest_volume_row = df.loc[df['volume'].idxmax()]
    highest_volume_date = highest_volume_row['timestamp'].strftime("%B %d, %Y")
    highest_volume_value = highest_volume_row['volume']

    latest_row = df.iloc[-1]
    latest_close = latest_row['close']
    latest_date = latest_row['timestamp'].strftime("%B %d, %Y")

    long_days = df[df['direction'] == 'LONG'].shape[0]
    short_days = df[df['direction'] == 'SHORT'].shape[0]
    neutral_days = df[df['direction'].isna()].shape[0]

    context = f"""
You are a financial assistant analyzing TSLA stock OHLCV data from a CSV.
Stats you should know:
- Highest volume spike: {highest_volume_value} on {highest_volume_date}.
- Latest closing price (as of {latest_date}): ${latest_close:.2f}.
- LONG days: {long_days}, SHORT days: {short_days}, Neutral: {neutral_days}.

Answer based on this data. Think like a market analyst.
"""
    return context

gemini_context = generate_gemini_context(df)

# Gemini query function
def ask_gemini(user_query):
    try:
        response = model.generate_content(gemini_context + "\n\nUser question: " + user_query)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# 🧠 Gemini Chatbot UI (in separate tab)
def gemini_chat_ui():
    st.title("🤖 Ask Questions About TSLA Stock Data")
    st.markdown("💬 Ask me anything about TSLA trends, support/resistance, or markers:")

    user_query = st.text_input("Type your question here", placeholder="e.g., How many LONG days in 2024?")

    if user_query:
        with st.spinner("Thinking..."):
            response = ask_gemini(user_query)
        st.success("Here's the answer:")
        st.markdown(response)

# --- Streamlit Tabs ---
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Gemini Chatbot"])

with tab2:
    gemini_chat_ui()

with st.sidebar:
    show_support = st.checkbox("Show Support Bands", True)
    show_resistance = st.checkbox("Show Resistance Bands", True)
    show_markers = st.checkbox("Show LONG/SHORT/Neutral Markers", True)

# Tabs: Clean separation
tab1, tab2, tab3 = st.tabs(["📊 All Elements", "🟩 Only Candles + Markers", "📉 Only Bands"])

def make_chart(show_all=True, only_markers=False, only_bands=False):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))

    # Markers
    if (show_all or only_markers) and show_markers:
        long_x, long_y = [], []
        short_x, short_y = [], []
        neutral_x, neutral_y = [], []

        for _, row in df.iterrows():
            if row['direction'] == 'LONG':
                long_x.append(row['timestamp'])
                long_y.append(row['low'] * 0.99)
            elif row['direction'] == 'SHORT':
                short_x.append(row['timestamp'])
                short_y.append(row['high'] * 1.01)
            else:
                neutral_x.append(row['timestamp'])
                neutral_y.append((row['high'] + row['low']) / 2)

        fig.add_trace(go.Scatter(x=long_x, y=long_y, mode='markers', marker=dict(color='green', symbol='arrow-up', size=9), name='LONG'))
        fig.add_trace(go.Scatter(x=short_x, y=short_y, mode='markers', marker=dict(color='red', symbol='arrow-down', size=9), name='SHORT'))
        fig.add_trace(go.Scatter(x=neutral_x, y=neutral_y, mode='markers', marker=dict(color='gold', symbol='circle', size=8), name='NEUTRAL'))

    # Bands
    if (show_all or only_bands):
        for _, row in df.iterrows():
            if show_support and row['Support']:
                fig.add_shape(type="rect",
                            x0=row['timestamp'], x1=row['timestamp'],
                            y0=min(row['Support']), y1=max(row['Support']),
                            fillcolor="rgba(0,255,0,0.2)", line_width=0, layer="below")
            if show_resistance and row['Resistance']:
                fig.add_shape(type="rect",
                            x0=row['timestamp'], x1=row['timestamp'],
                            y0=min(row['Resistance']), y1=max(row['Resistance']),
                            fillcolor="rgba(255,0,0,0.2)", line_width=0, layer="below")

    # Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=10)
    )

    return fig

# Render each tab
with tab1:
    st.plotly_chart(make_chart(show_all=True), use_container_width=True, key="chart_all")
with tab2:
    st.plotly_chart(make_chart(only_markers=True), use_container_width=True, key="chart_markers")
with tab3:
    st.plotly_chart(make_chart(only_bands=True), use_container_width=True, key="chart_bands")

# --- Step 3: Gemini Chatbot Integration ---

import google.generativeai as genai

# # 🔐 Enter your Gemini API key directly (safe for local apps)
# GEMINI_API_KEY = "AIzaSyCeV4yrnNLorjapsfkaMYgetgzcNwbM_Ug"  # Replace with your actual key

# genai.configure(api_key=GEMINI_API_KEY)

# # Create Gemini Pro model
# model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# st.subheader("🤖 Ask Questions About TSLA Stock Data")

# # Automatically summarize key insights
# latest_row = df.iloc[-1]
# summary_context = f"""
# You are an AI assistant analyzing Tesla (TSLA) stock data.

# The most recent date is: {latest_row['timestamp'].date()}.
# Price: Open={latest_row['open']}, High={latest_row['high']}, Low={latest_row['low']}, Close={latest_row['close']}.
# Trend direction on that date: {latest_row['direction']}.

# Support Levels: {latest_row['Support']}
# Resistance Levels: {latest_row['Resistance']}
# """

# # Chat UI
# user_input = st.text_input("💬 Ask me anything about TSLA trends, support/resistance, or markers:")

# if user_input:
#     with st.spinner("Gemini is thinking..."):
#         prompt = summary_context + "\n\nUser Query: " + user_input
#         response = model.generate_content(prompt)
#         st.success(response.text)
