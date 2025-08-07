import streamlit as st
import numpy as np
import cv2
import requests
from PIL import Image
from openai import OpenAI

# --- Page config ---
st.set_page_config(page_title="CRL Ultrasound Classifier + Chat", layout="wide")
st.title("🩻 Ultrasound Classifier + 🧠 CRL Q&A Assistant")

# --- Sidebar: API Key Input ---
st.sidebar.header("🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# --- Image Upload + Prediction ---
st.subheader("📷 Upload Ultrasound Image")
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Sending image to model..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("https://ultrasound-classification-webservice.onrender.com/predict/", files=files)
                response.raise_for_status()
                data = response.json()
                prediction = data["prediction"]
                probability = data["probability"]

                st.success(f"🧠 Prediction: **{prediction.upper()}**")
                st.write("📊 Raw Probability:", probability)

                # --- AI Interpretation Agent ---
                if api_key:
                    client = OpenAI(api_key=api_key)
                    agent_prompt = (
                        f"The uploaded ultrasound image was classified as **'{prediction}'** "
                        f"with a probability of **{probability:.2f}**.\n\n"
                        "Please provide a short, clinical-style explanation of this result. "
                        "Consider if the image might be usable for crown-rump length (CRL) measurement."
                    )
                    with st.spinner("Generating AI reasoning..."):
                        try:
                            agent_response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": agent_prompt}]
                            )
                            explanation = agent_response.choices[0].message.content
                            st.info(f"🧠 AI Interpretation:\n\n{explanation}")
                        except Exception as e:
                            st.warning(f"Could not generate AI interpretation: {e}")

            except Exception as e:
                st.error(f"Error contacting model API: {e}")

# --- Chatbot Section ---
st.subheader("💬 Ask about Crown-Rump Length (CRL)")

if api_key:
    client = OpenAI(api_key=api_key)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask a question about CRL or fetal growth...")

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.chat_history
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")
else:
    st.info("Enter your OpenAI API key in the sidebar to use the CRL chatbot.")
