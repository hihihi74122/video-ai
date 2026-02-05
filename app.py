import streamlit as st
import os
import cv2
import yt_dlp
from PIL import Image
import io
import base64
import uuid
import requests

# -----------------------------------------------------------------------------
# CONFIGURATION & SECRETS
# -----------------------------------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔒 Security Alert: HF_TOKEN not found.")
    st.stop()

# FINAL FIX: Using Idefics 2.
# 1. It is created by Hugging Face, so it is definitely on the free "Serverless" provider.
# 2. It supports Chat Completions (OpenAI format).
# 3. It is smarter than BLIP-2 and more reliable than LLaVA on the free tier.
MODEL_ID = "HuggingFaceM4/idefics2-8b-chatty"

API_URL = "https://router.huggingface.co/v1/chat/completions"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Idefics Video AI", layout="centered")
st.title("🤖 Idefics 2 Video AI")
st.markdown("Using HuggingFace's own Idefics 2 model for maximum stability.")

# -----------------------------------------------------------------------------
# VIDEO PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def download_video(url):
    unique_id = uuid.uuid4().hex
    filename = f"temp_video_{unique_id}.mp4"
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        if os.path.exists(filename):
            return filename
        return None
    except Exception as e:
        st.error(f"Download Failed: {str(e)}")
        return None

def extract_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
    finally:
        cap.release()
    return frames

def ask_ai_router(image, question):
    """
    Sends request to HuggingFace Router for Idefics 2.
    """
    # 1. Convert Image to Base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 2. Construct OpenAI-Style Payload
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                    {"type": "text", "text": question}
                ]
            }
        ],
        "max_tokens": 300
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            return "⏳ The AI is waking up. Wait 30 seconds and try again."
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()
        
        # Parse standard OpenAI Chat response format
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

num_frames = st.slider("Frames to analyze", 1, 16, 4)
video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is happening?")

if st.button("Analyze Video"):
    if not video_url or not user_question:
        st.warning("Please provide both a URL and a question.")
    else:
        video_file = None
        
        with st.status("Downloading video...", expanded=True) as status:
            video_file = download_video(video_url)
            if video_file:
                status.update(label="Download complete!", state="complete", expanded=False)
            else:
                status.update(label="Download failed.", state="error")
                st.stop()

        if video_file:
            with st.spinner(f"Extracting {num_frames} frames..."):
                frames = extract_frames(video_file, num_frames=num_frames)

                if not frames:
                    st.error("Could not read frames.")
                else:
                    st.subheader(f"Visual Context ({len(frames)} Frames)")
                    cols = st.columns(min(4, len(frames)))
                    for i, img in enumerate(frames):
                        col_idx = i % 4
                        cols[col_idx].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    # Analyze the middle frame
                    middle_frame_index = len(frames) // 2
                    
                    with st.spinner("AI is analyzing (Idefics 2)..."):
                        answer = ask_ai_router(frames[middle_frame_index], user_question)
                        
                        st.subheader("AI Answer")
                        st.info(answer)

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass
