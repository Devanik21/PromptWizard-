import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import os
from datetime import datetime
import pyperclip

st.set_page_config(page_title="Prompt Engineer's Toolkit", layout="centered")
st.title("🔧 Prompt Engineer's Toolkit")
st.markdown("""
Enter a **goal** and optionally a **poor prompt**. This tool will generate optimized prompt templates and debug the poor one.
""")

st.sidebar.title("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
model_option = st.sidebar.selectbox("Select Gemini Model", ["models/gemini-2.0-flash", "models/gemini-2.0-pro", "models/gemini-1.5-flash"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

if st.sidebar.checkbox("Enable Prompt History"):
    if not os.path.exists("prompt_history"):
        os.makedirs("prompt_history")

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

    goal = st.text_area("🎯 Your Goal", placeholder="E.g. Summarize academic research papers in a friendly tone", height=100)

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

    with st.expander("Advanced Options"):
        output_format = st.radio("Output Format", ["Text", "JSON", "Markdown"])
        num_prompts = st.slider("Number of prompts to generate", 1, 5, 3)

    tab1, tab2, tab3 = st.tabs(["Generate Prompts", "A/B Test Prompts", "💬 Prompt Coach Chat"])

    with tab1:
        if st.button("Generate Optimized Prompts") and goal:
            with st.spinner("Crafting prompt magic ✨"):
                prompt_format = output_format.lower()
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
                st.info(f"Response generated in {end_time - start_time:.2f} seconds")

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

                if st.button("Copy to Clipboard"):
                    pyperclip.copy(response.text)
                    st.success("Copied to clipboard!")

        with st.expander("📈 Prompt Quality Analyzer"):
            analyze_prompt = st.text_area("Enter a prompt to evaluate", height=100)
            if st.button("Analyze Quality"):
                score_prompt = f"""
You are an expert prompt engineer. Score the following prompt on:
- Clarity (0-10)
- Completeness (0-10)
- Tone (0-10)
- Effectiveness (0-10)
Then suggest improvements.
Prompt: {analyze_prompt}
"""
                analysis = model.generate_content(score_prompt)
                st.markdown(analysis.text)

        with st.expander("🧪 Prompt Sandbox"):
            sandbox_prompt = st.text_area("Prompt", height=100)
            sandbox_input = st.text_area("Input", height=100)
            if st.button("Run in Sandbox"):
                sandbox_result = model.generate_content(f"{sandbox_prompt}\n\n{sandbox_input}")
                st.markdown("### Result")
                st.markdown(sandbox_result.text)

        with st.expander("🎡 Generate Random Prompt Ideas"):
            idea_topic = st.text_input("Topic (e.g., marketing, education, writing)")
            if st.button("Get Prompt Ideas"):
                idea_response = model.generate_content(f"Generate 5 creative prompt ideas about: {idea_topic}")
                st.markdown(idea_response.text)

        with st.expander("🎨 Style Transformer"):
            base_prompt = st.text_area("Original Prompt")
            style = st.selectbox("Select Style", ["Friendly", "Professional", "Poetic", "Sarcastic"])
            if st.button("Transform Style"):
                styled_response = model.generate_content(f"Rewrite this prompt in a {style} tone:\n{base_prompt}")
                st.markdown(styled_response.text)

        with st.expander("📚 Prompt Template Library"):
            templates = {
                "Explain Code": "Explain the following code to a beginner:\n{code}",
                "Rewrite Article": "Rewrite this article for kids:\n{text}",
                "Keyword Extractor": "Extract main keywords from:\n{content}"
            }
            template_selected = st.selectbox("Choose Template", list(templates.keys()))
            st.code(templates[template_selected])

        with st.expander("🌍 Multilingual Prompt Helper"):
            original_prompt = st.text_area("Prompt in English")
            target_lang = st.selectbox("Translate to", ["Spanish", "French", "Hindi", "Japanese"])
            if st.button("Translate Prompt"):
                translated = model.generate_content(f"Translate to {target_lang}:\n{original_prompt}")
                st.markdown(translated.text)

        with st.expander("⚠️ Prompt Risk Checker"):
            check_prompt = st.text_area("Prompt to check", height=100)
            if st.button("Check Risk"):
                risk_result = model.generate_content(f"Analyze this prompt for sensitive content:\n{check_prompt}")
                st.markdown(risk_result.text)

        with st.expander("🧠 Prompt Use Case Generator"):
            theme_input = st.text_input("Enter a theme or domain")
            if st.button("Generate Use Cases"):
                use_cases = model.generate_content(f"Generate 5 prompt use cases for: {theme_input}")
                st.markdown(use_cases.text)

        with st.expander("📊 Prompt to Slide Generator"):
            slide_prompt = st.text_area("Enter topic for slides")
            if st.button("Generate Slides Outline"):
                slides = model.generate_content(f"Generate slide deck outline for: {slide_prompt}")
                st.markdown(slides.text)

        with st.expander("🧵 Thread Generator for X/Twitter"):
            thread_topic = st.text_input("Topic for Twitter Thread")
            if st.button("Generate Thread"):
                thread = model.generate_content(f"Generate a Twitter thread on: {thread_topic}")
                st.markdown(thread.text)

        with st.expander("🧩 Reverse Prompt Engineer"):
            reverse_prompt = st.text_area("Output you want", height=100)
            if st.button("Find Possible Prompt"):
                reverse = model.generate_content(f"What prompt would lead to this output:\n{reverse_prompt}")
                st.markdown(reverse.text)

        with st.expander("💡 Prompt Variants Explorer"):
            variant_base = st.text_area("Enter base prompt")
            if st.button("Generate Variants"):
                variants = model.generate_content(f"Generate 5 creative variants of: {variant_base}")
                st.markdown(variants.text)

        with st.expander("🔄 Prompt Reformatter"):
            messy_prompt = st.text_area("Enter poorly structured prompt")
            if st.button("Reformat Prompt"):
                cleaned = model.generate_content(f"Reformat this prompt clearly and professionally:\n{messy_prompt}")
                st.markdown(cleaned.text)

        with st.expander("🗃️ Prompt Storage Suggestor"):
            storage_type = st.text_input("Describe your use case")
            if st.button("Suggest Storage"):
                suggestion = model.generate_content(f"Suggest best way to store prompts for this use: {storage_type}")
                st.markdown(suggestion.text)

        with st.expander("🔍 SEO Prompt Optimizer"):
            seo_prompt = st.text_area("Enter content-related prompt")
            if st.button("Optimize for SEO"):
                optimized = model.generate_content(f"Optimize this prompt for SEO content creation:\n{seo_prompt}")
                st.markdown(optimized.text)

        with st.expander("🧠 Few-Shot Prompt Creator"):
            task_desc = st.text_area("Task Description")
            examples_few = st.text_area("Examples (comma separated)")
            if st.button("Generate Few-Shot Prompt"):
                few_shot = model.generate_content(f"Create few-shot prompt for: {task_desc} using examples: {examples_few}")
                st.markdown(few_shot.text)


    

    with tab2:
        st.write("Compare two different prompts to see which performs better")
        prompt_a = st.text_area("Prompt A", height=100)
        prompt_b = st.text_area("Prompt B", height=100)
        test_input = st.text_area("Test Input", placeholder="Enter text to test these prompts on", height=100)

        if st.button("Run A/B Test") and prompt_a and prompt_b and test_input:
            with st.spinner("Testing prompts..."):
                response_a = model.generate_content(f"{prompt_a}\n\n{test_input}")
                response_b = model.generate_content(f"{prompt_b}\n\n{test_input}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Result A")
                    st.markdown(response_a.text)

                with col2:
                    st.subheader("Result B")
                    st.markdown(response_b.text)

                if st.button("Export Results"):
                    results = {
                        "prompt_a": prompt_a,
                        "prompt_b": prompt_b,
                        "test_input": test_input,
                        "result_a": response_a.text,
                        "result_b": response_b.text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    df = pd.DataFrame([results])
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    with tab3:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You:", key="chat_input")
        if st.button("Send"):
            st.session_state.chat_history.append(("You", user_input))
            coach_reply = model.generate_content(f"You are a prompt engineering coach. Help with: {user_input}")
            st.session_state.chat_history.append(("Coach", coach_reply.text))

        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")
else:
    st.warning("Please enter your Gemini API key in the sidebar.")
