# Livestock Lens – Ml driven Multi-Faceted Livestock Welfare System

An AI-powered, non-invasive dairy health monitoring system for early detection of
lameness and mastitis in cattle using low-cost RGB vision and deep learning.

This project integrates:
- A CNN + LSTM spatio-temporal gait analysis model for lameness detection
- A CNN-based image classification model for mastitis detection
- An interactive Streamlit dashboard for real-time diagnosis and visualization

Designed for **affordability, scalability, and rural farm deployment**.

## Modules

### 1. Lameness Detection
- Input: Cow walking video
- Key techniques: Pose estimation (DeepLabCut) + CNN + LSTM
- Output: Healthy / Lame1 / Lame2 + confidence score

### 2. Mastitis Detection
- Input: Udder/teat image
- Model: MobileNetV2 based CNN
- Output: Healthy / Mastitis + confidence score

### 3. Unified Dashboard
- File: `final_dashboard.py`
- Built using: Streamlit
- Allows selection between Lameness & Mastitis modules

## How to Run
pip install -r Lameness/requirements.txt</br>
pip install -r Mastitis/requirements.txt</br>
streamlit run final_dashboard.py</br>
