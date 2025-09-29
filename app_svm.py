# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:48:00 2025

@author: AINIL MARDIAH
"""

import os
import cv2
import numpy as np
import pickle
import streamlit as st
import face_recognition
from sklearn.svm import SVC

st.title("Face Recognition dengan SVM")
st.write("Upload gambar wajah atau ambil dari webcam untuk identifikasi.")

# Model dan Dataset
MODEL_FILE = "face_svm_model.pkl"
DATA_PATH = "C:/Users/FX506HC/Machine Learning/Face_Recog/dataset"

# Load Model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    st.success("Model SVM berhasil dimuat dari file")
else:
    
    #Train Model
    st.warning("Model belum ada, sedang training dari dataset...")
    X, y = [], []
    for person_name in os.listdir(DATA_PATH):
        person_folder = os.path.join(DATA_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))   # resize biar seragam
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                X.append(encodings[0])
                y.append(person_name)

    #pwmodelan SVM
    model = SVC(kernel="linear", probability=True)  
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    st.success("Model SVM berhasil ditraining & disimpan")

# Upload Foto
uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = model.predict([face_encoding])[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hasil Prediksi dengan SVM")

# Webcam Capture
st.write("Atau gunakan webcam:")

camera_input = st.camera_input("Ambil foto dari webcam")
if camera_input is not None:
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = model.predict([face_encoding])[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hasil Prediksi Webcam dengan SVM")
