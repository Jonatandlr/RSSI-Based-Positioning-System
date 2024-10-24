import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta al archivo de texto (ajusta la ruta si es necesario)
file_path = './src/utils/todosPuntos5.txt'

# Leer el archivo con espacios como delimitador
df = pd.read_csv(file_path,sep='\s+')

# llenar valores nan con el siguiente valor
df.fillna(method='bfill', inplace=True)

# Mostrar una muestra de los datos para verificar la carga correcta
print("Estructura del dataset:")
print(df.head())

# Separar las columnas 'Punto' y 'Iteracion' de las señales de las antenas
X = df[['Punto', 'Iteracion']]
y = df[['AntenaA', 'AntenaB', 'AntenaC', 'AntenaD']]

# Convertir 'Punto' en variables dummy si es categórica
X = pd.get_dummies(X, columns=['Punto'], drop_first=True)

# Realizar el split de 80% para entrenamiento y 20% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar los conjuntos de entrenamiento y prueba en archivos CSV si es necesario
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Guardar en archivos CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nDatos de entrenamiento guardados en 'train_data.csv'")
print("Datos de prueba guardados en 'test_data.csv'")
