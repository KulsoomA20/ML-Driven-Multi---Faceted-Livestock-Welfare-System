import os
import time
import glob
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from Lameness.extract_features import extract_features_for_video  # function

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Page setup

st.set_page_config(page_title="Livestock Health Dashboard", layout="wide")

# Inject custom CSS

st.markdown("""
    <style>
    [data-testid="stDataFrame"] thead th {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        color: #222 !important;
        background-color: #f2f2f2 !important;
    }
    [data-testid="stDataFrame"] tbody td {
        color: #111 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSuccess"] * {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)


# File paths for lameness model and CSVs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LAMENESS_DIR = os.path.join(ROOT_DIR, "Lameness")
MASTITIS_DIR = os.path.join(ROOT_DIR, "Mastitis")

LAMENESS_MODEL_PATH = os.path.join(LAMENESS_DIR, "lstm_model_tuned1.h5")
DLC_CSVS_DIR = os.path.join(LAMENESS_DIR, "CSV")
VIDEO_FRAMES_DIR = os.path.join(LAMENESS_DIR, "Frames")

TEMP_DIR = os.path.join(ROOT_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)



# Lameness classes and colors

LAMENESS_CLASSES = ['Healthy', 'Lame1', 'Lame2']
LAMENESS_COLOR_MAP = {'Healthy': "#4CD251", 'Lame1': "#F4BB52", 'Lame2': "#E3382F"}

# Mastitis colors

MASTITIS_COLOR_MAP = {"Mastitis": "#E63946", "Healthy": "#4CD251"}

# Session state for animation

if 'run_animation' not in st.session_state:
    st.session_state['run_animation'] = True
if 'current_video' not in st.session_state:
    st.session_state['current_video'] = None

# Helpers for lameness

def load_lstm_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Lameness model load error: {e}")
        return None

def get_sample_frames_from_folder(frames_folder, n=30):
    jpgs = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
    pngs = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    files = jpgs + pngs
    files = files[:n]
    images = []
    for fp in files:
        try:
            images.append(Image.open(fp))
        except:
            pass
    return images

def get_sample_frames_from_video(video_path, n=30):
    frames = []
    if OPENCV_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // n)
        idx = 0
        while len(frames) < n and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            idx += 1
        cap.release()
    else:
        try:
            import imageio
            reader = imageio.get_reader(video_path)
            total = reader.count_frames()
            step = max(1, total // n)
            for i in range(0, total, step):
                if len(frames) >= n: break
                img = reader.get_data(i)
                frames.append(Image.fromarray(img))
            reader.close()
        except:
            pass
    return frames

def prepare_features(features):
    df = pd.DataFrame([features]).drop(columns=['video'], errors='ignore')
    arr = df.values
    return arr.reshape((1, 1, arr.shape[1]))

def set_animation_state_to_run():
    st.session_state['run_animation'] = True

# Mastitis prediction

@st.cache_resource
def load_mastitis_model():
    return load_model(os.path.join(MASTITIS_DIR, "mastitis_model.keras"))

def predict_mastitis(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    probs = load_mastitis_model().predict(x)[0]
    mastitis_prob = float(probs[0])
    healthy_prob = 1 - mastitis_prob
    if mastitis_prob > 0.5:
        label = "Mastitis"
        conf = mastitis_prob
    else:
        label = "Healthy"
        conf = healthy_prob
    return label, conf, {"Mastitis": mastitis_prob, "Healthy": healthy_prob}

# Lameness Dashboard

def lameness_dashboard():
    st.header("Cow Lameness Dashboard")
    uploaded_video = st.file_uploader("Upload a cow walking video", type=["mp4", "avi", "mov"])
    if uploaded_video is None: return

    video_filename = uploaded_video.name
    video_base = os.path.splitext(video_filename)[0]
    video_path = os.path.join(TEMP_DIR, video_filename)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Load LSTM model
    model = load_lstm_model(LAMENESS_MODEL_PATH)
    if model is None: return

    # Show video preview
    st.subheader(f"Video Preview — {video_filename}")
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.video(video_path)

    # Step 1: Frames
    st.markdown("## 1. Extracted Frames")
    frames_folder = os.path.join(VIDEO_FRAMES_DIR, video_base)
    frames = get_sample_frames_from_folder(frames_folder)
    if not frames:
        frames = get_sample_frames_from_video(video_path)

    if frames:
        col_left, col_animation, col_right = st.columns([1, 8, 1])
        with col_animation:
            img_placeholder = st.empty()
            if st.session_state['run_animation']:
                for i, frame in enumerate(frames):
                    img_placeholder.image(frame, use_container_width=True, caption=f"Frame {i+1}/{len(frames)}")
                    time.sleep(0.1)
                st.session_state['run_animation'] = False
            else:
                img_placeholder.image(frames[-1], use_container_width=True, caption="Finished animation")
        st.button("Replay", key="replay_anim", on_click=set_animation_state_to_run)

    # Step 2: CSV → LSTM
    st.markdown("## 2. Pose CSV → LSTM Input")
    csv_path = os.path.join(DLC_CSVS_DIR, f"{video_base}.csv")
    if not os.path.exists(csv_path):
        st.warning(f"CSV for {video_base} not found.")
        return
    features = extract_features_for_video(csv_path)
    X = prepare_features(features)
    st.success("Pose features prepared.")

    # Step 3: Classification
    st.markdown("## 3. Classification Result")
    preds = model.predict(X)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = LAMENESS_CLASSES[pred_idx]
    confidence = preds[pred_idx]

    # Prediction Card
    card_color = LAMENESS_COLOR_MAP.get(pred_label, "#3B96E6")
    st.markdown(f"""
        <div style="
            background-color: {card_color};
            padding: 18px;
            border-radius: 10px;
            color: white;
            font-weight: 800;
            font-size: 1.6rem;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        ">
            Predicted Lameness: {pred_label} — {confidence*100:.2f}% confidence
        </div>
    """, unsafe_allow_html=True)

    # Probability Table
    prob_df = pd.DataFrame({"Class": LAMENESS_CLASSES, "Probability": [float(p) for p in preds]})
    prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    st.dataframe(prob_df, use_container_width=True)

    # Cleanup
    try: os.remove(video_path)
    except: pass

# Mastitis Dashboard

def mastitis_dashboard():
    st.header("Mastitis Detection Dashboard")
    uploaded_file = st.file_uploader("Upload udder/teat image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None: return

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("1. Image Uploaded")
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.subheader("2. Preprocessing ")
        st.info("Resizing, scaling, and preparing image...")
        st.subheader("3. Classification")

        pred_label, confidence, prob_dict = predict_mastitis(uploaded_file)

        # Prediction Card
        card_color = MASTITIS_COLOR_MAP.get(pred_label, "#3B96E6")
        st.markdown(f"""
            <div style="
                background-color: {card_color};
                padding: 18px;
                border-radius: 10px;
                color: white;
                font-weight: 800;
                font-size: 1.4rem;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            ">
                Prediction: {pred_label} — {confidence*100:.2f}% confidence
            </div>
        """, unsafe_allow_html=True)

        # Probabilities Table
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Class", "Probability"])
        prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x*100:.2f}%")
        st.table(prob_df)

# Main selection

choice = st.radio("Select Dashboard", ["Cow Lameness", "Mastitis Detection"])
if choice == "Cow Lameness":
    lameness_dashboard()
else:
    mastitis_dashboard()
