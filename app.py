import streamlit as st
import os
import cv2
import yt_dlp
from huggingface_hub import InferenceClient
from PIL import Image
import io
import base64
import uuid
import time

# -----------------------------------------------------------------------------
# CONFIGURATION & SECRETS
# -----------------------------------------------------------------------------
# We securely load the token from Streamlit Secrets.
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔒 Security Alert: HF_TOKEN not found.")
    st.info("Please go to Streamlit Cloud -> Settings -> Secrets and add HF_TOKEN.")
    st.stop()

# Using a reliable open-source Vision Model
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Robust Video AI", layout="centered")
st.title("👁️ Video AI Analyzer")
st.markdown("Paste a link. The AI extracts frames and answers your question.")

# Cache the client to improve performance
@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN)

client = get_client()

# -----------------------------------------------------------------------------
# VIDEO PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def download_video(url):
    """
    Downloads video with robust options to avoid simple blocks.
    Uses a unique filename to prevent conflicts between users.
    """
    # Create a unique filename so multiple users don't overwrite each other
    unique_id = uuid.uuid4().hex
    filename = f"temp_video_{unique_id}.mp4"
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True, # Ensure we only get the single video
        # Simulate a browser to avoid some 403 errors
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        
        if os.path.exists(filename):
            return filename
        else:
            return None
    except Exception as e:
        st.error(f"Download Failed: {str(e)}")
        st.info("Tip: TikTok/Instagram sometimes block downloads. Try a YouTube link.")
        return None

def extract_frames(video_path, num_frames=4):
    """
    Extracts frames safely using a try-finally block to ensure
    the video file is closed before we try to delete it later.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        # Pick 4 evenly distributed frames
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB (Standard)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
    finally:
        # CRITICAL FIX: Always release the file lock
        cap.release()
    
    return frames

def ask_ai(image, question):
    """
    Sends image (as base64) and text to HuggingFace Chat Completion API.
    """
    # 1. Convert Image to Base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 2. Construct the message payload
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": question}
            ]
        }
    ]

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Inference Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is happening in this video?")

if st.button("Analyze Video"):
    if not video_url or not user_question:
        st.warning("Please provide both a URL and a question.")
    else:
        video_file = None
        
        # --- STEP 1: DOWNLOAD ---
        with st.status("Downloading video...", expanded=True) as status:
            video_file = download_video(video_url)
            if video_file:
                status.update(label="Download complete!", state="complete", expanded=False)
            else:
                status.update(label="Download failed.", state="error")
                st.stop()

        # --- STEP 2: EXTRACT FRAMES ---
        if video_file:
            with st.spinner("Extracting frames..."):
                frames = extract_frames(video_file)

                if not frames:
                    st.error("Could not read frames from the video. The file might be corrupt or an unsupported format.")
                else:
                    # Display Frames
                    st.subheader("Visual Context")
                    cols = st.columns(len(frames))
                    for i, img in enumerate(frames):
                        cols[i].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    # --- STEP 3: AI ANALYSIS ---
                    with st.spinner("AI is analyzing the frames..."):
                        # Analyze the first frame (or you could loop through all for better context)
                        answer = ask_ai(frames[0], user_question)
                        
                        st.subheader("AI Answer")
                        st.info(answer)

            # --- STEP 4: CLEANUP ---
            # Ensure file exists and try to remove it
            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except Exception as e:
                # If cleanup fails, log it but don't crash the app
                print(f"Cleanup error: {e}")



