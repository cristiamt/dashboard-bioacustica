import os
import sys
import json
import datetime
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURAÇÕES PRINCIPAIS ---

PASTA_IMAGENS_PARA_ANALISE = r"D:\Meu Drive\Python\Mastera\Bioacustica\Dataset\Teste\Teeeste" 

# Nome do ficheiro que guardará os tempos acumulados
NOME_FICHEIRO_DADOS = "dados_acumulados.json"

# Configurações do modelo
CLASSES = ["agua", "ocio", "pastejo", "ruminacao"]
IMG_SIZE = (128, 128)
SEGUNDOS_POR_IMAGEM = 5

# --- FUNÇÕES AUXILIARES ---

def carregar_ou_criar_dados():
    """Procura pelo ficheiro JSON. Se existir, carrega os dados. Se não, cria um novo com valores a zero."""
    try:
        with open(NOME_FICHEIRO_DADOS, 'r') as f:
            print(f"Ficheiro '{NOME_FICHEIRO_DADOS}' encontrado. Carregando totais...")
            return json.load(f)
    except FileNotFoundError:
        print(f"Ficheiro '{NOME_FICHEIRO_DADOS}' não encontrado. Criando um novo...")
        dados_novos = {
            "tempo_agua": 0,
            "tempo_ocio": 0,
            "tempo_pastejo": 0,
            "tempo_ruminacao": 0
        }
        salvar_dados(dados_novos)
        return dados_novos

def salvar_dados(dados):
    """Salva o dicionário de dados no ficheiro JSON."""
    with open(NOME_FICHEIRO_DADOS, 'w') as f:
        json.dump(dados, f, indent=4)

def formatar_tempo(segundos):
    """Converte segundos para um formato HH:MM:SS mais legível."""
    return str(datetime.timedelta(seconds=segundos))

def resetar_contagem():
    """Zera todos os contadores de tempo no ficheiro de dados."""
    print("⚠️  A zerar a contagem de tempo...")
    dados_zerados = {
        "tempo_agua": 0,
        "tempo_ocio": 0,
        "tempo_pastejo": 0,
        "tempo_ruminacao": 0
    }
    salvar_dados(dados_zerados)
    print("✅ Contagem zerada com sucesso!")

def verificar_regras(dados):
    """Verifica os tempos acumulados e RETORNA uma lista de alertas."""
    alertas = [] # Cria uma lista vazia para guardar as mensagens
    
    horas_pastejo = dados["tempo_pastejo"] / 3600
    if horas_pastejo < 8: alertas.append("❗️ Alerta de Pastejo: O gado está pastando pouco.")
    elif horas_pastejo > 12: alertas.append("❗️ Alerta de Pastejo: O gado está pastando muito.")

    horas_ruminacao = dados["tempo_ruminacao"] / 3600
    if horas_ruminacao < 6: alertas.append("❗️ Alerta de Ruminação: O gado está ruminando pouco.")
    elif horas_ruminacao > 10: alertas.append("❗️ Alerta de Ruminação: O gado está ruminando muito.")
    
    horas_ocio = dados["tempo_ocio"] / 3600
    if horas_ocio < 6: alertas.append("❗️ Alerta de Descanso: O gado está descansando pouco.")
    elif horas_ocio > 11: alertas.append("❗️ Alerta de Descanso: O gado está descansando muito.")
    
    return alertas # Retorna a lista de mensagens

# --- FUNÇÃO PRINCIPAL DE PREVISÃO ---

def identificar_atividade(caminho_da_imagem, modelo):
    """Usa o modelo de ML para prever a atividade numa imagem."""
    img = Image.open(caminho_da_imagem).convert("L").resize(IMG_SIZE)
    img_array = np.array(img)
    img_rgb = np.stack([img_array]*3, axis=-1)
    img_input = img_rgb.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3) / 255.0
    
    pred_probs = modelo.predict(img_input, verbose=0)[0]
    
    indice_da_classe = np.argmax(pred_probs)
    nome_da_classe = CLASSES[indice_da_classe]
    return nome_da_classe

# --- BLOCO DE EXECUÇÃO ---

if __name__ == "__main__":
    # Verifica se um "comando especial" foi usado
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'reset':
        resetar_contagem()
    else:
        # Modo de Análise em Lote
        print("--- Iniciando Análise de Comportamento em Lote ---")
        
        # Carrega o modelo de ML (apenas uma vez)
        print("Carregando modelo de IA... (pode demorar um pouco)")
        modelo_cnn = tf.keras.models.load_model("modelo_espectrograma.h5")
        
        # Carrega os dados de tempo
        dados_atuais = carregar_ou_criar_dados()
        
        # Lista as imagens na pasta de análise
        imagens_para_analisar = [f for f in os.listdir(PASTA_IMAGENS_PARA_ANALISE) if f.endswith('.png')]
        print(f"Encontradas {len(imagens_para_analisar)} imagens para analisar.")

        # Loop para processar cada imagem
        for i, nome_imagem in enumerate(imagens_para_analisar):
            caminho_completo = os.path.join(PASTA_IMAGENS_PARA_ANALISE, nome_imagem)
            
            # Faz a previsão
            atividade = identificar_atividade(caminho_completo, modelo_cnn)
            print(f"\n({i+1}/{len(imagens_para_analisar)}) Imagem '{nome_imagem}' -> Atividade: {atividade.upper()}")
            
            # Acumula o tempo
            chave_tempo = f"tempo_{atividade}"
            if chave_tempo in dados_atuais:
                dados_atuais[chave_tempo] += SEGUNDOS_POR_IMAGEM
            
            # Verifica as regras e salva o progresso
            verificar_regras(dados_atuais)
            salvar_dados(dados_atuais)

        print("\n--- Análise Concluída ---")
        print("Resumo dos Tempos Acumulados:")
        for atividade, segundos in dados_atuais.items():
            print(f"  - {atividade.replace('_', ' ').title()}: {formatar_tempo(segundos)}")
        print("--------------------------")