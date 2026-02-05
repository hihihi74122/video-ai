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

# HARD LIMIT: The Router only allows 4 images per request for this model.
MAX_IMAGES = 4

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Video Focus", layout="centered")
st.title("🎯 Smart Video Focus")
st.markdown("Due to API limits, we analyze 4 consecutive frames from a specific section.")

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

def extract_frames_by_interval(video_path, interval_seconds, max_total_frames=20):
    """
    Extracts up to 20 frames total.
    We extract more than 4 so we can choose the 'best' 4 later.
    """
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
            
            if len(frames) >= max_total_frames:
                break
            
            current_frame += frame_skip_float
    finally:
        cap.release()
    return frames

def ask_ai_sequence(images, question):
    """
    Sends exactly 4 images (or less) to the API.
    """
    content = [
        {
            "type": "text", 
            "text": f"These are 4 sequential frames from a video. Analyze the movement and action in this short clip to answer: {question}"
        }
    ]
    
    # Add images
    for img in images:
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

st.info("ℹ️ **Limit:** The API only allows analyzing 4 images at once.")

# 1. User selects which part of the video to watch
focus_section = st.selectbox(
    "Which part of the video should the AI watch?",
    ("End of Video (Conclusion)", "Start of Video (Introduction)", "Middle of Video")
)

# 2. Interval setting
interval = st.slider("Snapshot Interval (seconds)", 0.1, 5.0, 0.5, step=0.1)

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is happening in this part of the video?")

if st.button("Analyze Section"):
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
            # 1. Extract all potential frames
            with st.spinner(f"Extracting frames..."):
                all_frames = extract_frames_by_interval(video_file, interval_seconds=interval)

            if not all_frames:
                st.error("Could not read frames.")
            else:
                # 2. Select the 4 frames based on user choice
                selected_frames = []
                
                if focus_section == "End of Video (Conclusion)":
                    # Take last 4 frames
                    selected_frames = all_frames[-MAX_IMAGES:]
                elif focus_section == "Start of Video (Introduction)":
                    # Take first 4 frames
                    selected_frames = all_frames[:MAX_IMAGES]
                else:
                    # Take middle 4 frames
                    mid_index = len(all_frames) // 2
                    selected_frames = all_frames[mid_index-2 : mid_index+2]
                
                # Fallback if video is too short
                if len(selected_frames) == 0:
                    selected_frames = all_frames

                # 3. Display
                st.subheader(f"Visual Context ({len(selected_frames)} Frames - {focus_section})")
                cols = st.columns(len(selected_frames))
                for i, img in enumerate(selected_frames):
                    cols[i].image(img, caption=f"Frame {i+1}", use_column_width=True)

                # 4. Analyze
                with st.spinner("AI is analyzing this section..."):
                    answer = ask_ai_sequence(selected_frames, user_question)
                    
                    st.subheader("AI Answer")
                    st.info(answer)

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass
