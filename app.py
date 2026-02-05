import streamlit as st
import os
import cv2
import yt_dlp
from huggingface_hub import InferenceClient
from PIL import Image
import io

# --- CONFIGURATION ---
# We securely load the token from Streamlit Secrets.
try:
    # In Streamlit Cloud, this reads from the "Secrets" menu.
    # Locally, you can test this by creating a file .streamlit/secrets.toml with: HF_TOKEN = "your_token"
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔒 Security Alert: HF_TOKEN not found.")
    st.info("If you are the developer, please add your Hugging Face Token in the Streamlit Cloud 'Secrets' menu.")
    st.stop()

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# --- SETUP ---
st.set_page_config(page_title="Secure Video AI", layout="centered")
st.title("👁️ Secure Video AI")
st.write("Paste a link below. The AI will visually analyze the video frames.")

# Initialize Client (Cached to avoid reloading on every interaction)
@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN)

client = get_client()

# --- HELPER FUNCTIONS ---

def download_video(url):
    """Downloads video to local container storage."""
    # Define output template
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': 'video_temp.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        # We limit download size slightly to prevent hitting quotas too fast
        'max_filesize': 50 * 1024 * 1024, # 50MB limit
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get extension
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None

def extract_frames(video_path, num_frames=4):
    """Extracts N frames evenly spaced throughout the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file. It might be a corrupted format.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Calculate indices to grab
    if total_frames > 0:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
    
    cap.release()
    return frames

def ask_ai_about_image(image, question):
    """
    Sends an image and question to the Hugging Face Model.
    Uses the specific prompt format for LLaVA models.
    """
    # Convert PIL image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # LLaVA specific prompt format
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    try:
        # We use text_generation with the images parameter
        response = client.text_generation(
            model=MODEL_ID,
            prompt=prompt,
            images=[image_bytes], 
            max_new_tokens=200
        )
        return response
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- MAIN APP UI ---

video_url = st.text_input("Video URL (YouTube, TikTok, etc.):", placeholder="https://www.youtube.com/watch?v=...")
user_question = st.text_input("What do you want to know?", placeholder="Describe the person in the video.")

if st.button("Analyze"):
    if not video_url or not user_question:
        st.warning("Please provide both a link and a question.")
    else:
        # 1. Download
        with st.spinner("Downloading video (this may take a moment)..."):
            video_file = download_video(video_url)
        
        if video_file and os.path.exists(video_file):
            # 2. Extract Frames
            with st.spinner("Processing video frames..."):
                frames = extract_frames(video_file)
                
                if frames:
                    # Show the user what the AI sees
                    st.subheader("What the AI sees:")
                    cols = st.columns(len(frames))
                    for i, img in enumerate(frames):
                        cols[i].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    # 3. Analyze (We analyze the middle frame for best context in this free version)
                    with st.spinner("Thinking..."):
                        # Analyze the first frame to get an immediate answer
                        answer = ask_ai_about_image(frames[0], user_question)
                        
                        st.subheader("AI Answer:")
                        st.success(answer)
                else:
                    st.error("Could not extract frames from this video.")

            # 4. Cleanup (Delete the video file to save space on the cloud server)
            try:
                os.remove(video_file)
            except:
                pass