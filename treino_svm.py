import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

st.title("🎯 Classificador SVM para Espectrogramas de Bovinos 🐄")

# Caminhos
base_path = r"C:\Users\Chericoni\Bioacustica\Imagens"
train_dir = os.path.join(base_path, "Treinob")
test_dir = os.path.join(base_path, "Testeb")
img_size = (128, 128)

# Função de carregamento
def carregar_dados(pasta_base):
    X, y = [], []
    classes = sorted(os.listdir(pasta_base))
    for classe in classes:
        classe_path = os.path.join(pasta_base, classe)
        for arquivo in os.listdir(classe_path):
            if arquivo.endswith(".png"):
                caminho = os.path.join(classe_path, arquivo)
                img = Image.open(caminho).convert("L").resize(img_size)
                X.append(np.array(img).flatten())
                y.append(classe)
    return np.array(X), np.array(y)

if st.button("🚀 Treinar modelo SVM"):
    with st.spinner("🔹 Carregando dados de treino..."):
        X_train, y_train = carregar_dados(train_dir)

    with st.spinner("🔹 Carregando dados de teste..."):
        X_test, y_test = carregar_dados(test_dir)

    with st.spinner("🔧 Treinando modelo SVM..."):
        modelo = SVC(kernel='rbf', C=10, gamma='scale')
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    # Acurácia e F1
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    st.success(f"✅ Acurácia: {acc:.4f}")
    st.info(f"🎯 F1-score macro: {f1_macro:.4f} | micro: {f1_micro:.4f}")

    # Relatório
    st.subheader("📋 Métricas por Classe")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    st.dataframe(df_report.style.format("{:.2f}"))

    # Gráfico de barras
    st.subheader("📊 F1-score por classe")
    fig_f1, ax = plt.subplots()
    df_report["f1-score"].plot(kind="bar", ax=ax)
    ax.set_title("F1-score por classe")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    st.pyplot(fig_f1)

    # Matriz de Confusão
    st.subheader("🔍 Matriz de Confusão")
    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax_cm)
    ax_cm.set_xlabel("Predito")
    ax_cm.set_ylabel("Real")
    st.pyplot(fig_cm)

    #salva modelo
    modelo_path = "modelo_svm.pkl"
    joblib.dump(modelo, modelo_path)
    st.success(f"💾 Modelo SVM salvo como: {modelo_path}")
    