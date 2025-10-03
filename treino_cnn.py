import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.title("🐄 Treinamento do Classificador de Espectrogramas")

# Caminhos
base_path = r"D:\Meu Drive\Python\Mastera\Bioacustica\Dataset"
train_dir = os.path.join(base_path, "Treino")
test_dir = os.path.join(base_path, "Teste")

# Parâmetros
img_height, img_width = 128, 128
batch_size = st.sidebar.slider("Batch size", 8, 64, 16)
epochs = st.sidebar.slider("Épocas de treino", 1, 50, 10)

# Gatilho de treino
if st.button("🚀 Iniciar Treinamento"):
    with st.spinner("Treinando..."):
        # Geradores
        train_gen = ImageDataGenerator(rescale=1./255)
        test_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_data = test_gen.flow_from_directory(
            test_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Modelo CNN
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Treinamento
        # Containers do Streamlit para atualizar em tempo real
        acc_chart = st.line_chart()
        loss_chart = st.line_chart()
        
        # Callback para atualizar os gráficos
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.acc_values = []
                self.loss_values = []
        
            def on_epoch_end(self, epoch, logs=None):
                acc = logs.get('accuracy')
                val_acc = logs.get('val_accuracy')
                loss = logs.get('loss')
                val_loss = logs.get('val_loss')
        
                self.acc_values.append({'Treino': acc, 'Validação': val_acc})
                self.loss_values.append({'Treino': loss, 'Validação': val_loss})
        
                acc_chart.add_rows(pd.DataFrame(self.acc_values[-1], index=[epoch]))
                loss_chart.add_rows(pd.DataFrame(self.loss_values[-1], index=[epoch]))

                st.info(f"📈 Época {epoch+1}/{epochs} — Acurácia: {acc:.4f} — Val.: {val_acc:.4f}")

        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            callbacks=[StreamlitCallback()],
            verbose=0
        )

        
        st.success("✅ Treinamento concluído!")

        # Plot Acurácia e Loss
        st.subheader("📈 Curvas de Acurácia e Perda")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history['accuracy'], label='Treino')
        ax[0].plot(history.history['val_accuracy'], label='Validação')
        ax[0].set_title("Acurácia")
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Treino')
        ax[1].plot(history.history['val_loss'], label='Validação')
        ax[1].set_title("Perda")
        ax[1].legend()

        st.pyplot(fig)

        # Avaliação
        y_pred_probs = model.predict(test_data)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_data.classes
        class_names = list(test_data.class_indices.keys())

        st.subheader("📋 Relatório de Classificação")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Matriz de Confusão
        st.subheader("🔍 Matriz de Confusão")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_xlabel("Predito")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

        # Salvar modelo
        caminho_modelo = "modelo_espectrograma.h5"
        model.save(caminho_modelo)
        st.success(f"📁 Modelo salvo como: {caminho_modelo}")
