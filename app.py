# LLM_Playground/app.py

import streamlit as st
import google.generativeai as genai
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Playground",
    page_icon="🤖",
    layout="wide"
)

# --- Application Title ---
st.title("🤖 Conversational LLM Web App")
st.caption("Powered by Google Gemini")

# --- Gemini API Configuration ---
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Gemini API key not found. Please add it to your Streamlit secrets.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# --- Helper function to yield text from the stream ---
def stream_gemini_response(stream):
    """Yields text chunks from the Gemini API stream."""
    for chunk in stream:
        if chunk.text:
            yield chunk.text

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    st.header("System Prompt")
    system_prompt = st.text_area(
        "Define the bot's persona and rules.",
        "You are a friendly and helpful assistant.",
        label_visibility="collapsed"
    )

    # --- MODIFIED: Model is now hardcoded ---
    selected_model = "gemini-2.5-flash"
    st.info(f"Using model: **{selected_model}**") # Display the model being used
    
    st.header("Parameter Controls")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.75, 0.05)
    max_tokens = st.slider("Max Tokens", 50, 4096, 512, 8)

    st.divider()

    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

    def format_chat_history_for_export(messages):
        formatted_text = ""
        for message in messages:
            role = "You" if message["role"] == "user" else "Bot"
            formatted_text += f"{role}:\n{message['content']}\n\n"
        return formatted_text.strip()

    if "messages" in st.session_state and st.session_state.messages:
        chat_export_data = format_chat_history_for_export(st.session_state.messages)
        st.download_button(
            label="Export Conversation",
            data=chat_export_data,
            file_name=f"conversation_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain"
        )

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages from simplified history
for message in st.session_state.messages:
    role = "assistant" if message["role"] == "model" else "user"
    avatar = "🤖" if role == "assistant" else "🧑‍💻"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("Enter your message here..."):
    # Store and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Bot is thinking..."):
            try:
                model = genai.GenerativeModel(
                    selected_model,
                    system_instruction=system_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                
                gemini_history = [{"role": m["role"], "parts": [m["content"]]} for m in st.session_state.messages]
                
                # Use st.write_stream to correctly display the output
                response_stream = model.generate_content(
                    gemini_history,
                    stream=True
                )
                
                full_response = st.write_stream(stream_gemini_response(response_stream))
                
                # Store the final, complete response in our simple history
                st.session_state.messages.append({"role": "model", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")