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

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="True Video AI", layout="centered")
st.title("👁️ True Video AI")
st.markdown("Now analyzes a **sequence** of frames to understand movement.")

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

def extract_frames_by_interval(video_path, interval_seconds, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: return []
        
        frame_skip_float = fps * interval_seconds
        current_frame = 0.0
        
        while True:
            frame_idx = int(current_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            if len(frames) >= max_frames:
                break
            
            current_frame += frame_skip_float
    finally:
        cap.release()
    return frames

def ask_ai_sequence(images, question):
    """
    FIX: Sends MULTIPLE images to the AI to simulate video.
    We send the last N images (e.g., 8) so the AI can see the transition.
    """
    
    # 1. Prepare the Content List
    content = [
        {
            "type": "text", 
            "text": f"These are sequential frames from a video. Analyze the movement, action, and changes across these frames. Answer this question: {question}"
        }
    ]
    
    # 2. Add images to the content
    # We limit to last 8 images to prevent API token/size errors while still seeing motion
    images_to_send = images[-8:] if len(images) > 8 else images
    
    for img in images_to_send:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
        )
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # Increased timeout for multiple images
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 503:
            return "⏳ AI is loading... wait 1 minute."
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

interval = st.slider("Snapshot Interval (seconds)", 0.1, 5.0, 1.0, step=0.1)
st.caption("Shorter interval = more frames. The AI analyzes the last sequence of frames.")

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What action is happening in the video?")

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
                    cols = st.columns(4)
                    for i, img in enumerate(frames):
                        col_idx = i % 4
                        cols[col_idx].image(img, caption=f"{i*interval:.1f}s", use_column_width=True)

                    # Analyze the SEQUENCE
                    with st.spinner(f"AI analyzing the last {min(8, len(frames))} frames for movement..."):
                        answer = ask_ai_sequence(frames, user_question)
                        
                        st.subheader("AI Answer (Motion Analysis)")
                        st.info(answer)

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass
