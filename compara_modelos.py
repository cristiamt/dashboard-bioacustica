import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

st.title("🔍 Classificador de Espectrogramas - CNN vs SVM")

# Modelos disponíveis
modelos = {
    "CNN (Keras)": "modelo_espectrograma.h5",
    "SVM (scikit-learn)": "modelo_svm.pkl"
}

modelo_escolhido = st.selectbox("Escolha o modelo:", list(modelos.keys()))

# Upload de imagem
imagem = st.file_uploader("Envie um espectrograma .png", type=["png"])

if imagem:
    st.image(imagem, caption="Imagem enviada", use_column_width=True)

    # Pré-processamento da imagem
    img = Image.open(imagem).convert("L").resize((128, 128))
    img_array = np.array(img)

    if modelo_escolhido == "CNN (Keras)":
        modelo = load_model(modelos[modelo_escolhido])
        img_rgb = np.stack([img_array]*3, axis=-1)
        img_input = img_rgb.reshape(1, 128, 128, 3) / 255.0
        pred_probs = modelo.predict(img_input)[0]
        classe_predita = np.argmax(pred_probs)
        confianca = np.max(pred_probs)

        # Labels assumidos (ordem baseada no flow_from_directory)
        labels = ["agua", "ocio", "pastejo", "ruminacao"]
        st.success(f"📌 Classe predita (CNN): **{labels[classe_predita]}** com {confianca:.2%} de confiança")

    elif modelo_escolhido == "SVM (scikit-learn)":
        modelo = joblib.load(modelos[modelo_escolhido])
        vetor = img_array.flatten().reshape(1, -1)
        classe_predita = modelo.predict(vetor)[0]
        st.success(f"📌 Classe predita (SVM): **{classe_predita}**")
    
