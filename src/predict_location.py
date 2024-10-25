import joblib
import pandas as pd
# Cargar el modelo y el escalador cuando los necesites
modelo_cargado = joblib.load('modelo_entrenado.pkl')
scaler_cargado = joblib.load('scaler.pkl')

# Convertir la nueva muestra en un DataFrame con las mismas columnas que se usaron en el entrenamiento
nueva_muestra = pd.DataFrame([[-57,-75,-54,-47]], columns=['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD'])
# Escalar la nueva muestra
nueva_muestra_escalada = scaler_cargado.transform(nueva_muestra)
# Hacer la predicción
prediccion = modelo_cargado.predict(nueva_muestra_escalada)

print(f'Predicción del modelo cargado: {prediccion}')