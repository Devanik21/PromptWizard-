# app.py
import streamlit as st
import google.generativeai as genai

# --- App UI ---
st.set_page_config(page_title="Prompt Engineer's Toolkit", layout="centered")
st.title("🔧 Prompt Engineer’s Toolkit")
st.markdown("""
Enter a **goal** and optionally a **poor prompt**. This tool will generate optimized prompt templates and debug the poor one.
""")

# --- Sidebar for API Key ---
st.sidebar.title("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # --- Inputs ---
    goal = st.text_area("🎯 Your Goal", placeholder="E.g. Summarize academic research papers in a friendly tone", height=100)
    poor_prompt = st.text_area("🐌 Poor Prompt (optional)", placeholder="E.g. summarize this", height=100)

    if st.button("Generate Optimized Prompts") and goal:
        with st.spinner("Crafting prompt magic ✨"):
            prompt = f"""
You are an expert prompt engineer. Given the goal:
'{goal}'

1. Suggest 3 optimized prompts that would achieve this goal.
2. If a poor prompt is provided, critique it and rewrite it.

Poor Prompt (if any): '{poor_prompt}'
"""

            response = model.generate_content(prompt)
            st.subheader("🧠 Gemini's Output")
            st.markdown(response.text)
    else:
        st.info("Please enter your goal to generate prompts.")
else:
    st.warning("Please enter your Gemini API key in the sidebar.")
