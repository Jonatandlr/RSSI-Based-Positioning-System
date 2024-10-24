import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Cargar los datos ya separados (entrenamiento y prueba)
train_data = pd.read_csv('train_data.csv')  # Actualiza con la ruta del archivo de entrenamiento
test_data = pd.read_csv('test_data.csv')    # Actualiza con la ruta del archivo de prueba

# Características (intensidades de antenas) y etiquetas (puntos)
# Aquí seleccionamos las columnas que contienen las intensidades y las etiquetas
X_train = train_data[['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD']]
y_train = train_data.iloc[:, 1:21].idxmax(axis=1)  # Aquí seleccionamos las columnas de los puntos para entrenar

X_test = test_data[['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD']]
y_test = test_data.iloc[:, 1:21].idxmax(axis=1)  # Similar a lo anterior, para el conjunto de prueba

# Escalar los datos (opcional, pero recomendado para modelos de regresión)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train_scaled, y_train)

# Hacer predicciones en los datos de prueba
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy * 100:.2f}%')

# Reporte de clasificación
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", cm)

# Función para predecir un nuevo conjunto de intensidades de antena
def predecir_punto(antena_a, antena_b, antena_c, antena_d):
    nueva_muestra = np.array([[antena_a, antena_b, antena_c, antena_d]])
    nueva_muestra_escalada = scaler.transform(nueva_muestra)
    prediccion = model.predict(nueva_muestra_escalada)
    return prediccion[0]

# Ejemplo de predicción con nuevas intensidades de antena
resultado = predecir_punto(-59, -76, -40, -42)
print(f'El modelo predice que el punto está en: {resultado}')
