import pandas as pd
import matplotlib.pyplot as plt
import os

# Criar diretórios para salvar imagens
os.makedirs("imagem/normal", exist_ok=True)
os.makedirs("imagem/anomaly", exist_ok=True)

# Função para gerar imagens a partir de um DataFrame
def gerar_imagens(df, tipo, offset=0):
    for i in range(len(df)):
        signal = df.iloc[i, :-1]  # dados do ECG
        label = df.iloc[i, -1]    # classe (0 = normal, 1 = anormal)

        plt.figure(figsize=(2, 2))
        plt.plot(signal, color='black')
        plt.axis('off')

        filename = f"imagem/{'normal' if label == 0 else 'anomaly'}/{i + offset}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

# Carregar CSVs
df_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)
df_normal = pd.read_csv("ptbdb_normal.csv", header=None)

# Gerar imagens para cada classe
gerar_imagens(df_abnormal, tipo="anomaly", offset=0)
gerar_imagens(df_normal, tipo="normal", offset=len(df_abnormal))