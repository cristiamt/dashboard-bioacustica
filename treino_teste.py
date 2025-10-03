import os
import shutil
import random
import unicodedata

# Caminho dos arquivos de entrada
entrada = r"C:\Users\Chericoni\Bioacustica\Imagens"
saida_base = r"C:\Users\Chericoni\Bioacustica\Imagens"

# Proporção treino/teste
proporcao_treino = 0.7

# Classes esperadas
classes = ["pastejo", "ruminacao", "ocio", "agua"]

# Função para normalizar rótulo (sem acento, minúsculo)
def extrair_classe(nome_arquivo):
    nome = unicodedata.normalize("NFD", nome_arquivo.lower())
    nome = nome.encode("ascii", "ignore").decode("utf-8")
    for classe in classes:
        if classe in nome:
            return classe
    return None

# Coletar arquivos por classe
arquivos_por_classe = {classe: [] for classe in classes}
for arquivo in os.listdir(entrada):
    if arquivo.endswith(".png"):
        classe = extrair_classe(arquivo)
        if classe:
            arquivos_por_classe[classe].append(arquivo)

# Criar pastas e mover arquivos
for classe, arquivos in arquivos_por_classe.items():
    random.shuffle(arquivos)
    n_total = len(arquivos)
    n_treino = int(n_total * proporcao_treino)
    treino, teste = arquivos[:n_treino], arquivos[n_treino:]

    for tipo, lista in [("Treino", treino), ("Teste", teste)]:
        destino_classe = os.path.join(saida_base, tipo, classe)
        os.makedirs(destino_classe, exist_ok=True)
        for arquivo in lista:
            origem = os.path.join(entrada, arquivo)
            destino = os.path.join(destino_classe, arquivo)
            shutil.copy2(origem, destino)

        print(f"{classe.upper()} - {tipo}: {len(lista)} arquivos")

print("\n✅ Organização concluída!")
