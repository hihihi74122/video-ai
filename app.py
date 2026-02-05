import streamlit as st
import os
import cv2
import yt_dlp
from huggingface_hub import InferenceClient
from PIL import Image
import io
import base64

# --- CONFIGURATION ---
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔒 Security Alert: HF_TOKEN not found.")
    st.info("Please add your Hugging Face Token in the Streamlit Cloud 'Secrets' menu.")
    st.stop()

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# --- SETUP ---
st.set_page_config(page_title="Secure Video AI", layout="centered")
st.title("👁️ Secure Video AI (Fixed)")
st.write("Paste a link below. The AI will visually analyze the video frames.")

@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN)

client = get_client()

# --- HELPER FUNCTIONS ---

def download_video(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': 'video_temp.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'max_filesize': 50 * 1024 * 1024, 
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None

def extract_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
    cap.release()
    return frames

def ask_ai_about_image(image, question):
    """
    FIXED FUNCTION: Uses chat_completion with base64 image encoding.
    """
    # 1. Convert PIL Image to Base64 String
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 2. Prepare the message payload
    # This format mimics the OpenAI chat format which HuggingFace accepts
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": question}
            ]
        }
    ]

    try:
        # 3. Use chat_completion instead of text_generation
        completion = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=200
        )
        return completion.choices[0].message.content
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
        with st.spinner("Downloading video..."):
            # FIX: Changed 'url' to 'video_url' here
            video_file = download_video(video_url)
        
        if video_file and os.path.exists(video_file):
            # 2. Extract Frames
            with st.spinner("Processing video frames..."):
                frames = extract_frames(video_file)
                
                if frames:
                    st.subheader("What the AI sees:")
                    cols = st.columns(len(frames))
                    for i, img in enumerate(frames):
                        cols[i].image(img, caption=f"Frame {i+1}", use_column_width=True)

                    # 3. Analyze
                    with st.spinner("Thinking..."):
                        # We analyze the first frame for this demo
                        answer = ask_ai_about_image(frames[0], user_question)
                        
                        st.subheader("AI Answer:")
                        st.success(answer)
                else:
                    st.error("Could not extract frames from this video.")

            # 4. Cleanup
            try:
                os.remove(video_file)
            except:
                pass
