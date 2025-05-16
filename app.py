with tab3:
            st.subheader("📈 Prompt Analytics")
            
            # --- New Feature 33: Analytics Dashboard ---
            metrics_row = st.columns(4)
            
            with metrics_row[0]:
                st.metric(label="🧮 Current Session", value=st.session_state.prompt_count)
            
            with metrics_row[1]:
                session_time = time.time() - st.session_state.session_start
                minutes, seconds = divmod(int(session_time), 60)
                st.metric(label="⏱️ Session Duration", value=f"{minutes}m {seconds}s")
            
            # History metrics
            prompt_history_files = []
            if os.path.exists("prompt_history"):
                prompt_history_files = [f for f in os.listdir("prompt_history") if f.endswith(".json")]
            
            with metrics_row[2]:
                st.metric(label="📚 Total Saved Prompts", value=len(prompt_history_files))
            
            with metrics_row[3]:
                # Count distinct dates
                if prompt_history_files:
                    dates = set()
                    for f in prompt_history_files:
                        date_part = f.split("_")[1]  # Extract date from filename format
                        dates.add(date_part)
                    st.metric(label="📅 Active Days", value=len(dates))
                else:
                    st.metric(label="📅 Active Days", value=0)
            
            # --- New Feature 34: Usage Trends ---
            if prompt_history_files:
                st.subheader("📊 Usage Trends")
                
                # Parse dates from filenames
                dates = []
                for f in prompt_history_files:
                    try:
                        date_str = f.split("_")[1]
                        dates.append(datetime.strptime(date_str, "%Y%m%d").date())
                    except:
                        continue
                
                # Count prompts per day
                date_counts = {}
                for date in dates:
                    if date in date_counts:
                        date_counts[date] += 1
                    else:
                        date_counts[date] = 1
                
                # Create dataframe for plotting
                df = pd.DataFrame({
                    'date': list(date_counts.keys()),
                    'count': list(date_counts.values())
                })
                
                # Sort by date
                df = df.sort_values('date')
                
                # Plot the trend
                fig = px.line(df, x='date', y='count', title='Prompts Generated Over Time')
                fig.update_layout(xaxis_title='Date', yaxis_title='Number of Prompts')
                st.plotly_chart(fig, use_container_width=True)
                
                # --- New Feature 35: Prompt Theme Analysis ---
                st.subheader("🔍 Prompt Theme Analysis")
                
                # Extract themes from prompt history
                themes = []
                for f in prompt_history_files[:50]:  # Limit to 50 files for performance
                    try:
                        with open(f"prompt_history/{f}", "r") as file:
                            data = json.load(file)
                            if "goal" in data:
                                themes.append(data["goal"])
                    except:
                        continue
                
                if themes:
                    # Create word cloud
                    all_text = " ".join(themes)
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        colormap='viridis').generate(all_text)
                    
                    # Display word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
            else:
                st.info("ℹ️ No analytics data available yet. Generate some prompts first!")
                
            # --- New Feature 36: A/B Test Results Analysis ---
            if os.path.exists("ab_tests") and len(os.listdir("ab_tests")) > 0:
                st.subheader("🧪 A/B Test Results Analysis")
                
                # Load A/B test data
                ab_test_files = [f for f in os.listdir("ab_tests") if f.endswith(".json")]
                winner_counts = {"A": 0, "B": 0, "Equal": 0}
                
                for f in ab_test_files:
                    try:
                        with open(f"ab_tests/{f}", "r") as file:
                            data = json.load(file)
                            if "winner" in data:
                                winner_counts[data["winner"]] += 1
                    except:
                        continue
                
                # Create bar chart of winners
                fig = px.bar(
                    x=list(winner_counts.keys()),
                    y=list(winner_counts.values()),
                    title="A/B Test Winners",
                    labels={'x': 'Winner', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import os
import re
import nltk
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pyperclip
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import base64
from gtts import gTTS

# --- App UI ---
st.set_page_config(page_title="Prompt Engineer's Toolkit", layout="wide")
st.title("🔧 Prompt Engineer's Toolkit")
st.markdown("""
Enter a **goal** and optionally a **poor prompt**. This tool will generate optimized prompt templates and debug the poor one.
""")

# --- New Feature 11: Custom CSS ---
st.markdown("""
<style>
    .main-header {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .feature-button {background-color: #ff4b4b; color: white; padding: 10px;}
    .success-box {background-color: #d4edda; padding: 10px; border-radius: 5px;}
    .info-box {background-color: #d1ecf1; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar for API Key ---
with st.sidebar:
    st.title("🔐 API Configuration")
    
    # --- New Feature 12: User Profiles ---
    profiles = ["Default", "New Profile"]
    if os.path.exists("user_profiles"):
        saved_profiles = [f.replace(".json", "") for f in os.listdir("user_profiles") if f.endswith(".json")]
        profiles = ["Default", *saved_profiles, "New Profile"]
    
    selected_profile = st.selectbox("👤 User Profile", profiles)
    
    if selected_profile == "New Profile":
        new_profile_name = st.text_input("Enter profile name")
        if st.button("Create Profile") and new_profile_name:
            if not os.path.exists("user_profiles"):
                os.makedirs("user_profiles")
            with open(f"user_profiles/{new_profile_name}.json", "w") as f:
                json.dump({"api_key": "", "preferred_model": "models/gemini-2.0-flash"}, f)
            st.success(f"Profile {new_profile_name} created!")
            st.experimental_rerun()
    
    profile_data = {"api_key": "", "preferred_model": "models/gemini-2.0-flash"}
    if selected_profile != "Default" and selected_profile != "New Profile":
        with open(f"user_profiles/{selected_profile}.json", "r") as f:
            profile_data = json.load(f)
    
    api_key = st.text_input("🔑 Enter your Gemini API Key", value=profile_data.get("api_key", ""), type="password")
    
    # --- New Feature 13: API Key Management ---
    if st.checkbox("💾 Save API key to profile") and selected_profile != "Default" and selected_profile != "New Profile":
        profile_data["api_key"] = api_key
        with open(f"user_profiles/{selected_profile}.json", "w") as f:
            json.dump(profile_data, f)
        st.success("API key saved to profile!")

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
    generation_config = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_output_tokens": max_tokens
    }
    model = genai.GenerativeModel(model_option, generation_config=generation_config)
    
    # --- New Feature 18: Multiple View Modes ---
    view_mode = st.radio("👓 View Mode", ["Standard", "Split View", "Compact"], horizontal=True)
    
    if view_mode == "Split View":
        col1, col2 = st.columns(2)
        with col1:
            main_container = st
        with col2:
            output_container = st
    else:
        main_container = st
        output_container = st
        
    # --- Inputs ---
    with main_container:
        # --- New Feature 19: Goal Templates with Icons ---
        goal_templates = {
            "": "🔍 Custom Goal",
            "academic_summary": "📚 Summarize academic research papers in a friendly tone",
            "data_analysis": "📊 Create detailed data analysis prompts for complex datasets",
            "creative_story": "🎭 Generate creative short stories based on a theme",
            "code_explanation": "💻 Explain complex code in simple terms for beginners",
            "marketing_copy": "📣 Create persuasive marketing copy for products",
            "interview_questions": "🎯 Generate interview questions for a specific role",
            "email_template": "📧 Create professional email templates",
            "seo_optimization": "🔍 Optimize content for search engines"
        }
        
        template_choice = st.selectbox("📋 Choose a template or create your own:", options=list(goal_templates.keys()))
        goal_placeholder = goal_templates.get(template_choice, "🔍 Custom Goal").split(" ", 1)[1] if template_choice else "E.g. Summarize academic research papers in a friendly tone"
        
        # --- New Feature 20: Rich Text Editor ---
        st.markdown("### 🎯 Your Goal")
        st.markdown("""
        <div style="border:1px solid #ccc; border-radius:5px; padding:10px;">
        <small>Use <b>bold</b>, *italics*, `code`, etc. (Markdown supported)</small>
        </div>
        """, unsafe_allow_html=True)
        
        goal = st.text_area("", placeholder=goal_placeholder, height=100, key="goal_input")
        if template_choice and not goal:
            goal = goal_templates.get(template_choice, "").split(" ", 1)[1]
    
            # --- New Feature 21: Poor Prompt Enhancement ---
        st.markdown("### 🐌 Poor Prompt (optional)")
        
        poor_prompt_col1, poor_prompt_col2 = st.columns([3, 1])
        with poor_prompt_col1:
            poor_prompt = st.text_area("", placeholder="E.g. summarize this", height=100, key="poor_prompt")
        with poor_prompt_col2:
            if st.button("🔄 Auto-Fix"):
                if poor_prompt:
                    with st.spinner("Analyzing prompt..."):
                        fix_response = model.generate_content(f"Identify the main issues with this prompt and fix it: '{poor_prompt}'")
                        poor_prompt = fix_response.text
                        st.session_state.poor_prompt = poor_prompt
                else:
                    st.warning("⚠️ Please enter a prompt to fix")
        
        # --- New Feature 22: Context Upload ---
        st.markdown("### 📄 Context (optional)")
        context_file = st.file_uploader("🔼 Upload a file for context", type=["txt", "pdf", "docx", "md"])
        context_text = ""
        if context_file is not None:
            context_text = context_file.getvalue().decode("utf-8")
            st.success(f"✅ File uploaded: {context_file.name} ({len(context_text)} characters)")
            
        # --- New Feature 23: Prompt Tags ---
        st.markdown("### 🏷️ Prompt Tags (optional)")
        tags_input = st.text_input("Add tags separated by commas")
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
        if tags:
            st.write("Tags:", " ".join([f"**#{tag}**" for tag in tags if tag]))
            
        # --- New Feature 24: Prompt Complexity Analyzer ---
        if st.button("🔍 Analyze Prompt Complexity"):
            if goal:
                with st.spinner("Analyzing complexity..."):
                    # Simple complexity analysis
                    words = len(re.findall(r'\w+', goal))
                    sentences = len(re.findall(r'[.!?]+', goal)) + 1
                    complexity_score = 0
                    
                    if words < 10:
                        complexity = "Basic"
                        complexity_score = 1
                    elif words < 30:
                        complexity = "Moderate"
                        complexity_score = 2
                    else:
                        complexity = "Advanced"
                        complexity_score = 3
                    
                    # Check for advanced prompt engineering techniques
                    techniques = []
                    if "step by step" in goal.lower():
                        techniques.append("Step-by-step reasoning")
                    if "example" in goal.lower():
                        techniques.append("Examples")
                    if "format:" in goal.lower() or "output:" in goal.lower():
                        techniques.append("Output formatting")
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = complexity_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Prompt Complexity"},
                        gauge = {
                            'axis': {'range': [0, 3]},
                            'steps': [
                                {'range': [0, 1], 'color': "lightgray"},
                                {'range': [1, 2], 'color': "gray"},
                                {'range': [2, 3], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': complexity_score
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"**Complexity:** {complexity}")
                    st.write(f"**Word count:** {words}")
                    st.write(f"**Sentence count:** {sentences}")
                    if techniques:
                        st.write(f"**Techniques detected:** {', '.join(techniques)}")
            else:
                st.warning("⚠️ Please enter a prompt to analyze")
    
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
        with tab4:
            st.subheader("🔊 Text-to-Speech Generator")
            
            # --- New Feature 37: Text-to-Speech Converter ---
            tts_text = st.text_area("✏️ Enter text to convert to speech", height=150)
            
            tts_options_col1, tts_options_col2 = st.columns(2)
            
            with tts_options_col1:
                tts_language = st.selectbox("🌐 Language", 
                                          ["English", "Spanish", "French", "German", "Japanese", "Chinese"])
                
                language_codes = {
                    "English": "en", "Spanish": "es", "French": "fr",
                    "German": "de", "Japanese": "ja", "Chinese": "zh-CN"
                }
                
            with tts_options_col2:
                tts_speed = st.slider("⚡ Speech Speed", 0.5, 1.5, 1.0, 0.1)
            
            if st.button("🎙️ Generate Speech") and tts_text:
                with st.spinner("Generating audio..."):
                    try:
                        tts = gTTS(
                            text=tts_text,
                            lang=language_codes.get(tts_language, "en"),
                            slow=(tts_speed < 0.8)
                        )
                        audio_file = BytesIO()
                        tts.save(audio_file)
                        audio_file.seek(0)
                        audio_bytes = audio_file.read()
                        
                        st.audio(audio_bytes, format="audio/mp3")
                        
                        # --- New Feature 38: Audio Download ---
                        st.download_button(
                            label="💾 Download Audio",
                            data=audio_bytes,
                            file_name=f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                            mime="audio/mp3"
                        )
                    except Exception as e:
                        st.error(f"Error generating speech: {str(e)}")
        
        # --- New Feature 39: Quick Tips Sidebar ---
        with st.sidebar.expander("💡 Prompt Engineering Tips"):
            st.markdown("""
            **Top Tips:**
            - 🎯 Be specific about your desired outcome
            - 📋 Include examples of what you want
            - 🧩 Break complex prompts into steps
            - 🔄 Use "step by step" for reasoning tasks
            - 📏 Specify output format when needed
            - 🚫 Include what NOT to do
            """)
        
        # --- New Feature 40: System Status Indicator ---
        with st.sidebar:
            st.markdown("---")
            st.subheader("🖥️ System Status")
            
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                st.markdown("API Status:")
            with status_col2:
                st.markdown("✅ Operational")
                
            if st.button("🔄 Check API Status"):
                with st.spinner("Checking..."):
                    try:
                        # Simple API check with minimal token usage
                        test_response = model.generate_content("Hello")
                        if test_response:
                            st.success("✅ API is responding correctly!")
                        else:
                            st.error("❌ API returned empty response")
                    except Exception as e:
                        st.error(f"❌ API Error: {str(e)}")
                        
            # Session info
            st.markdown(f"**Session ID:** {st.session_state.session_id[:8]}...")
            st.markdown(f"**Generated prompts:** {st.session_state.prompt_count}")
else:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
