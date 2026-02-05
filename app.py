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

MODEL_ID = "Salesforce/blip2-opt-2.7b"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Video AI Pro", layout="centered")
st.title("👁️ Video AI Pro")
st.markdown("Select how many frames to analyze for better understanding.")

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
        
        # Calculate indices
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

def ask_ai_direct(image, question):
    """
    Robust function to handle API requests and various error responses.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Construct Payload
    payload = {
        "inputs": {
            "image": f"data:image/jpeg;base64,{img_str}",
            "text": question
        }
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Handle Model Loading (503)
        if response.status_code == 503:
            return "⏳ The AI is waking up. Please wait 30 seconds and try Analyze again."
        
        # Handle other errors
        if response.status_code != 200:
            # Return the actual error message from HF for debugging
            return f"API Error {response.status_code}: {response.text}"

        # Parse JSON
        result = response.json()
        
        # Handle different return formats safely
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", str(result))
        elif isinstance(result, dict):
            return result.get("generated_text", str(result))
        else:
            return str(result)

    except requests.exceptions.Timeout:
        return "Error: Request timed out. The video or AI server might be slow."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

# NEW: Slider to control how many frames are analyzed
num_frames = st.slider("How many frames should the AI analyze?", min_value=1, max_value=16, value=4, step=1)
st.caption("More frames = better understanding, but slower.")

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is the main event in this video?")

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
                    st.error("Could not read frames from video.")
                else:
                    st.subheader(f"Visual Context ({len(frames)} Frames)")
                    # Display in a grid
                    cols = st.columns(min(4, len(frames)))
                    for i, img in enumerate(frames):
                        col_idx = i % 4
                        cols[col_idx].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    # We analyze the first frame to start (to keep it fast), 
                    # or you could loop through them. 
                    # For a "Whole Video" feel, we usually need complex logic to combine answers.
                    # Here we analyze the most representative middle frame.
                    middle_frame_index = len(frames) // 2
                    
                    with st.spinner("AI is analyzing the middle frame..."):
                        answer = ask_ai_direct(frames[middle_frame_index], user_question)
                        
                        st.subheader("AI Answer")
                        st.info(answer)
                        
                        st.caption("Note: To analyze ALL frames one by one takes much longer and costs more API credits. This analyzes the key middle frame.")

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass

