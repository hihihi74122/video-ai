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

# FINAL MODEL: Qwen2-VL-7B-Instruct
# This is a top-tier open-source model hosted directly by Hugging Face.
# It is robust, supports the Chat format, and works on the free tier.
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

API_URL = "https://router.huggingface.co/v1/chat/completions"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Qwen Video AI", layout="centered")
st.title("👀 Qwen Video AI")
st.markdown("Using Qwen2-VL. Adjust the interval to capture more of the video.")

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

def extract_frames_by_interval(video_path, interval_seconds=5):
    """
    Extracts frames based on time interval (e.g., every 2 seconds).
    This gives a better sense of 'watching' than just 4 static frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        if not cap.isOpened():
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return []

        # Calculate how many frames to skip to match the interval
        frame_skip = int(fps * interval_seconds)
        
        current_frame = 0
        count = 0
        
        # Safety limit: Don't extract more than 20 frames to prevent crashing UI/API
        max_frames = 20 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                count += 1
                if count >= max_frames:
                    break
            
            current_frame += 1
            
    finally:
        cap.release()
    
    return frames

def ask_ai_qwen(image, question):
    """
    Sends request to Router for Qwen2-VL.
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
            # Show detailed error for debugging
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

# Slider to control how often we take a snapshot (Interval in seconds)
# Lower = more frames (slower), Higher = fewer frames (faster)
interval = st.slider("Snapshot Interval (seconds)", 1, 10, 3)
st.caption(f"The AI will take a photo every {interval} seconds (Max 20 photos).")

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is happening in the video?")

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
            with st.spinner(f"Extracting frames every {interval} seconds..."):
                frames = extract_frames_by_interval(video_file, interval_seconds=interval)

                if not frames:
                    st.error("Could not read frames.")
                else:
                    st.subheader(f"Visual Context ({len(frames)} Snapshots)")
                    # Display in a grid
                    cols = st.columns(4)
                    for i, img in enumerate(frames):
                        col_idx = i % 4
                        cols[col_idx].image(img, caption=f"T+{i*interval}s", use_column_width=True)

                    # Analyze the LAST frame (often the conclusion/result of the video)
                    last_frame = frames[-1]
                    
                    with st.spinner("AI is analyzing the final snapshot..."):
                        answer = ask_ai_qwen(last_frame, user_question)
                        
                        st.subheader("AI Answer")
                        st.info(answer)

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass
