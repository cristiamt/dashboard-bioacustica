import streamlit as st
import os
import tensorflow as tf
from PIL import Image

# Importa as funções e constantes do nosso script de backend
from analisador_bioacustico import (
    identificar_atividade, 
    verificar_regras, 
    formatar_tempo, 
    CLASSES, 
    SEGUNDOS_POR_IMAGEM
)

# --- Configuração da Página ---
st.set_page_config(layout="wide")

# --- Cabeçalho ---
st.title("🐄 Dashboard de Monitoramento de Atividade Bovina")
st.markdown("Faça o upload de um lote de imagens de espectrograma (.png) para análise.")
st.divider()

# --- Cache do Modelo ---
@st.cache_resource
def carregar_modelo_ia():
    """Carrega o modelo de ML e o guarda em cache para não recarregar a cada ação."""
    return tf.keras.models.load_model("modelo_espectrograma.h5")

modelo = carregar_modelo_ia()

# --- Interface de Upload ---
st.header("1. Faça o Upload do Lote de Imagens")
uploaded_files = st.file_uploader(
    "Escolha os ficheiros de espectrograma (.png)",
    type=["png"],
    accept_multiple_files=True
)

# --- Botão de Análise e Lógica Principal ---
st.header("2. Inicie a Análise")
if st.button("🚀 Analisar Lote de Imagens"):
    if uploaded_files:
        # Inicializa os contadores e a barra de progresso
        dados_acumulados = {f"tempo_{c}": 0 for c in CLASSES}
        progress_bar = st.progress(0, text="Análise em progresso...")
        
        # Loop através dos ficheiros enviados
        for i, uploaded_file in enumerate(uploaded_files):
            # A função de previsão precisa de um caminho, então salvamos a imagem temporariamente
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Chama o nosso "motor" para fazer a previsão
            atividade = identificar_atividade(uploaded_file.name, modelo)
            
            # Acumula o tempo
            chave_tempo = f"tempo_{atividade}"
            if chave_tempo in dados_acumulados:
                dados_acumulados[chave_tempo] += SEGUNDOS_POR_IMAGEM
            
            # Remove o ficheiro temporário
            os.remove(uploaded_file.name)
            
            # Atualiza a barra de progresso
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analisando: {uploaded_file.name}")

        progress_bar.empty() # Limpa a barra de progresso ao final
        st.success("✅ Análise em lote concluída!")
        st.divider()

        # --- Exibição do Relatório Final ---
        st.header("📈 Relatório Final de Atividade")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Tempos Totais Acumulados")
            for atividade_key, segundos in dados_acumulados.items():
                nome_bonito = atividade_key.replace("tempo_", "").capitalize()
                st.metric(label=nome_bonito, value=formatar_tempo(segundos))

        with col2:
            st.subheader("🔔 Alertas Gerados")
            alertas = verificar_regras(dados_acumulados)
            if alertas:
                for alerta in alertas:
                    st.warning(alerta)
            else:
                st.success("Nenhum alerta gerado. Comportamento dentro dos parâmetros normais.")

    else:
        st.warning("Por favor, faça o upload de pelo menos uma imagem antes de analisar.")