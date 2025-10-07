import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Caminhos das pastas
normal_path = './imagem/normal'
abnormal_path = './imagem/anomaly'
img_size = (64, 64)

# Função para carregar imagens e rotular
def load_images(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append((img, label))
    return data

# Carregando imagens
normal_data = load_images(normal_path, 0)
abnormal_data = load_images(abnormal_path, 1)

# Unindo e preparando os dados
all_data = normal_data + abnormal_data
X = np.array([x[0].flatten() for x in all_data], dtype='float32') / 255.0
y = np.array([x[1] for x in all_data])

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construindo o modelo MLP
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilando e treinando
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Avaliação
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\n✅ Acurácia no teste: {accuracy:.2f}')

# Visualização do desempenho
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()