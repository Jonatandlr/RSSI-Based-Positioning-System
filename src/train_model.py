import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar los datos ya separados (entrenamiento y prueba)
train_data = pd.read_csv('./DataSets/DataToTrain/Train/train_data.csv')  # Actualiza con la ruta del archivo de entrenamiento
test_data = pd.read_csv('./DataSets/DataToTrain/Test/test_data.csv')    # Actualiza con la ruta del archivo de prueba

# Características (intensidades de antenas) y etiquetas (puntos)
X_train = train_data[['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD']]
y_train = train_data.iloc[:, 1:21].idxmax(axis=1)  # Seleccionamos las columnas de los puntos para entrenar

X_test = test_data[['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD']]
y_test = test_data.iloc[:, 1:21].idxmax(axis=1)  # Similar a lo anterior, para el conjunto de prueba

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión logística
logistic_model = LogisticRegression(max_iter=1000)

# Aplicar OneVsRestClassifier sobre el modelo
model = OneVsRestClassifier(logistic_model)
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
# Guardar el modelo y el escalador
joblib.dump(model, 'modelo_entrenado.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modelo y escalador guardados exitosamente.")

# # Función para predecir un nuevo conjunto de intensidades de antena
# def predecir_punto(antena_a, antena_b, antena_c, antena_d):
#     # Convertir la nueva muestra a un DataFrame con los mismos nombres de columna que X_train
#     nueva_muestra = pd.DataFrame([[antena_a, antena_b, antena_c, antena_d]], columns=['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD'])
    
#     # Escalar la nueva muestra
#     nueva_muestra_escalada = scaler.transform(nueva_muestra)
    
#     # Hacer la predicción
#     prediccion = model.predict(nueva_muestra_escalada)
    
#     return prediccion[0]

# # Ejemplo de predicción con nuevas intensidades de antena
# resultado = predecir_punto(-59, -76, -40, -42)
# print(f'El modelo predice que el punto está en: {resultado}')

