import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import os
from datetime import datetime
import pyperclip

# --- App UI ---
st.set_page_config(page_title="Prompt Engineer's Toolkit", layout="centered")
st.title("🔧 Prompt Engineer's Toolkit")
st.markdown("""
Enter a **goal** and optionally a **poor prompt**. This tool will generate optimized prompt templates and debug the poor one.
""")

# --- Sidebar for API Key ---
st.sidebar.title("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# --- New Feature 1: Model Selection ---
model_option = st.sidebar.selectbox(
    "Select Gemini Model",
    ["models/gemini-2.0-flash", "models/gemini-2.0-pro", "models/gemini-1.5-flash"]
)

# --- New Feature 2: Temperature Setting ---
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

# --- New Feature 3: Save Prompts Feature ---
if st.sidebar.checkbox("Enable Prompt History"):
    if not os.path.exists("prompt_history"):
        os.makedirs("prompt_history")

# --- New Feature 4: Dark/Light Mode Toggle ---
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_option, generation_config={"temperature": temperature})
    
    # --- Inputs ---
    goal = st.text_area("🎯 Your Goal", placeholder="E.g. Summarize academic research papers in a friendly tone", height=100)
    
    # --- New Feature 5: Example Library Dropdown ---
    examples = {
        "": "Custom Goal",
        "academic_summary": "Summarize academic research papers in a friendly tone",
        "data_analysis": "Create detailed data analysis prompts for complex datasets",
        "creative_story": "Generate creative short stories based on a theme",
        "code_explanation": "Explain complex code in simple terms for beginners"
    }
    example_choice = st.selectbox("Or choose from example goals:", options=list(examples.keys()))
    if example_choice and example_choice != "":
        goal = examples[example_choice]
    
    poor_prompt = st.text_area("🐌 Poor Prompt (optional)", placeholder="E.g. summarize this", height=100)
    
    # --- New Feature 6: Advanced Options Expander ---
    with st.expander("Advanced Options"):
        output_format = st.radio("Output Format", ["Text", "JSON", "Markdown"])
        num_prompts = st.slider("Number of prompts to generate", 1, 5, 3)
    
    # --- New Feature 7: A/B Testing Tab ---
    tab1, tab2 = st.tabs(["Generate Prompts", "A/B Test Prompts"])
    
    with tab1:
        if st.button("Generate Optimized Prompts") and goal:
            with st.spinner("Crafting prompt magic ✨"):
                prompt_format = "text"
                if output_format == "JSON":
                    prompt_format = "json"
                
                prompt = f"""
You are an expert prompt engineer. Given the goal:
'{goal}'
1. Suggest {num_prompts} optimized prompts that would achieve this goal.
2. If a poor prompt is provided, critique it and rewrite it.
Poor Prompt (if any): '{poor_prompt}'
Please format your response in {prompt_format} format.
"""
                start_time = time.time()
                response = model.generate_content(prompt)
                end_time = time.time()
                
                st.subheader("🧠 Gemini's Output")
                st.markdown(response.text)
                
                # --- New Feature 8: Performance Metrics ---
                st.info(f"Response generated in {end_time - start_time:.2f} seconds")
                
                # Save prompt history if enabled
                if st.sidebar.checkbox("Enable Prompt History", key="save_history"):
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"prompt_history/prompt_{now}.json", "w") as f:
                        json.dump({
                            "goal": goal,
                            "poor_prompt": poor_prompt,
                            "response": response.text,
                            "model": model_option,
                            "temperature": temperature
                        }, f)
                
                # --- New Feature 9: Copy to Clipboard Button ---
                if st.button("Copy to Clipboard"):
                    pyperclip.copy(response.text)
                    st.success("Copied to clipboard!")
                
        else:
            st.info("Please enter your goal to generate prompts.")
    
    with tab2:
        st.write("Compare two different prompts to see which performs better")
        prompt_a = st.text_area("Prompt A", height=100)
        prompt_b = st.text_area("Prompt B", height=100)
        test_input = st.text_area("Test Input", placeholder="Enter text to test these prompts on", height=100)
        
        if st.button("Run A/B Test") and prompt_a and prompt_b and test_input:
            with st.spinner("Testing prompts..."):
                # Process prompt A
                response_a = model.generate_content(f"{prompt_a}\n\n{test_input}")
                
                # Process prompt B
                response_b = model.generate_content(f"{prompt_b}\n\n{test_input}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Result A")
                    st.markdown(response_a.text)
                
                with col2:
                    st.subheader("Result B")
                    st.markdown(response_b.text)
                
                # --- New Feature 10: Export Results ---
                if st.button("Export Results"):
                    results = {
                        "prompt_a": prompt_a,
                        "prompt_b": prompt_b,
                        "test_input": test_input,
                        "result_a": response_a.text,
                        "result_b": response_b.text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Create a DataFrame for easy download
                    df = pd.DataFrame([results])
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
else:
    st.warning("Please enter your Gemini API key in the sidebar.")
