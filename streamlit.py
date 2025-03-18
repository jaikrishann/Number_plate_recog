import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Load pre-trained model
harcascade = 'model/haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(harcascade)

# Create directory if not exists
if not os.path.exists('plates'):
    os.makedirs('plates')

st.set_page_config(page_title='License Plate Detection System', layout='wide')

# Initialize session state for capturing
if 'capture' not in st.session_state:
    st.session_state.capture = False
if 'count' not in st.session_state:
    st.session_state.count = 0

# Tabs for Realtime Detection and History
realtime_tab, history_tab = st.tabs(['📸 Real-Time Detection', '🗃️ Saved Plates History'])

with realtime_tab:
    st.title('📸 License Plate Detection System')
    st.sidebar.title('Controls')

    min_area = st.sidebar.slider('Minimum Area', 100, 10000, 500)

    start_detection = st.sidebar.button('Start Detection')
    stop_detection = st.sidebar.button('Stop Detection')

    if st.sidebar.button('Capture Plate'):
        st.session_state.capture = True

    FRAME_WINDOW = st.empty()
    status_text = st.sidebar.empty()

    cap = None

    if start_detection:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        status_text.info('📷 Camera Started!')

    if cap and cap.isOpened():
        while True:
            success, img = cap.read()
            if not success:
                st.write('Failed to access the camera. Make sure it is connected.')
                break

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    img_roi = img[y: y + h, x: x + w]

                    if st.session_state.capture:
                        plate_path = f'plates/scanned_img_{st.session_state.count}.jpg'
                        cv2.imwrite(plate_path, img_roi)
                        st.session_state.count += 1
                        status_text.success('✅ Plate Captured Successfully!')
                        st.session_state.capture = False  # Reset capture state

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(img_rgb, channels='RGB')

            if stop_detection:
                cap.release()
                status_text.warning('🛑 Camera Stopped.')
                break


with history_tab:
    st.title('🗃️ Saved Plates History')
    saved_files = [f'plates/{file}' for file in os.listdir('plates') if file.endswith('.jpg')]

    if saved_files:
        st.write('### Previously Saved Plates')
        for file in saved_files:
            st.image(file, width=150)
    else:
        st.write('No plates saved yet.')
