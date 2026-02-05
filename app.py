import streamlit as st
import os
import cv2
import yt_dlp
from huggingface_hub import InferenceClient
from PIL import Image
import uuid

# -----------------------------------------------------------------------------
# CONFIGURATION & SECRETS
# -----------------------------------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔒 Security Alert: HF_TOKEN not found.")
    st.stop()

# CHANGE LOG: Switched to BLIP-2.
# Reason: It is the most stable model on the free HuggingFace Inference API.
# It is not a "chat" model, so we use text_generation instead.
MODEL_ID = "Salesforce/blip2-opt-2.7b"

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Stable Video AI", layout="centered")
st.title("👁️ Stable Video AI (BLIP-2)")
st.markdown("Uses the reliable BLIP-2 model to analyze video frames.")

@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN)

client = get_client()

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

def ask_ai(image, question):
    """
    FIXED FUNCTION:
    BLIP-2 is NOT a chat model. It uses 'text_generation'.
    We pass the PIL image directly.
    """
    # BLIP-2 specific prompt format
    prompt = f"Question: {question} Answer:"
    
    try:
        # We use text_generation and pass the image directly
        response = client.text_generation(
            model=MODEL_ID,
            prompt=prompt,
            image=image,  # Passing the PIL image object directly
            max_new_tokens=100
        )
        return response
    except Exception as e:
        return f"AI Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

video_url = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("Question:", placeholder="What is in this video?")

if st.button("Analyze"):
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
            with st.spinner("Extracting frames..."):
                frames = extract_frames(video_file)

                if not frames:
                    st.error("Could not read frames.")
                else:
                    st.subheader("Visual Context")
                    cols = st.columns(len(frames))
                    for i, img in enumerate(frames):
                        cols[i].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    with st.spinner("AI is analyzing..."):
                        # Analyze first frame
                        answer = ask_ai(frames[0], user_question)
                        
                        st.subheader("AI Answer")
                        st.info(answer)

            try:
                if os.path.exists(video_file):
                    os.remove(video_file)
            except:
                pass
