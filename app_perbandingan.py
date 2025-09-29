# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:49:07 2025

@author: AINIL MARDIAH
"""

import os
import cv2
import numpy as np
import pickle
import streamlit as st
import face_recognition
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Face Recognition: Naïve Bayes vs SVM")
st.write("Bandingkan performa antara Naïve Bayes dan SVM untuk pengenalan wajah.")

# Load Model dan Dataset
NB_MODEL_FILE = "face_nb_model.pkl"
SVM_MODEL_FILE = "face_svm_model.pkl"
DATA_PATH = "C:/Users/FX506HC/Machine Learning/Face_Recog/dataset"

# Ekstrak Fitur
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
        img = cv2.resize(img, (256, 256))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) > 0:
            X.append(encodings[0])
            y.append(person_name)

X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Naïve Bayes
if os.path.exists(NB_MODEL_FILE):
    with open(NB_MODEL_FILE, "rb") as f:
        nb_model = pickle.load(f)
else:
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    with open(NB_MODEL_FILE, "wb") as f:
        pickle.dump(nb_model, f)

y_pred_nb = nb_model.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Model SVM
if os.path.exists(SVM_MODEL_FILE):
    with open(SVM_MODEL_FILE, "rb") as f:
        svm_model = pickle.load(f)
else:
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, y_train)
    with open(SVM_MODEL_FILE, "wb") as f:
        pickle.dump(svm_model, f)

y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Tampilkan Evaluasi
st.subheader("Evaluasi Akurasi")
st.write(f"**Naïve Bayes Accuracy:** {acc_nb:.2%}")
st.write(f"**SVM Accuracy:** {acc_svm:.2%}")

# Pilih Model untuk Prediksi
model_choice = st.selectbox("Pilih model untuk prediksi:", ["Naïve Bayes", "SVM"])
model = nb_model if model_choice == "Naïve Bayes" else svm_model

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

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Hasil Prediksi dengan {model_choice}")

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

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Hasil Prediksi Webcam dengan {model_choice}")
