import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Caminhos para as pastas de imagens
normal_path = './imagem/normal'
abnormal_path = './imagem/anomaly'

# Listas para armazenar imagens e rótulos
images = []
labels = []

# Tamanho desejado para as imagens
img_size = (64, 64)

# Função para carregar imagens de uma pasta e rotulá-las
def load_images(folder, label):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Converte para tons de cinza
            if img is not None:
                img = cv2.resize(img, img_size)  # Redimensiona para 128x128
                images.append(img)
                labels.append(label)

# Carrega as imagens normais (rótulo 0)
load_images(normal_path, 0)

# Carrega as imagens anormais (rótulo 1)
load_images(abnormal_path, 1)

# Converte listas para arrays numpy
X = np.array(images, dtype='float32') / 255.0  # Normaliza para [0, 1]
y = np.array(labels)

# Remodela as imagens para vetores (MLP exige entrada vetorizada)
X = X.reshape(len(X), -1)

# Verificações
print("Total de imagens:", len(X))
print("Formato de X (dados):", X.shape)
print("Formato de y (rótulos):", y.shape)

# Exibe a primeira imagem como exemplo
plt.imshow(images[0], cmap='gray')
plt.title(f"Exemplo de imagem - Classe: {labels[0]}")
plt.axis('off')
plt.show()
