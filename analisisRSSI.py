import os
import pandas as pd

# Ruta de la carpeta donde están los archivos .txt
folder_path = r'C:\Users\Andrés Xicará\Documents\Archivos Python Data Analysis\Archivos txt'

# Inicializar un diccionario para almacenar los datos
data = {'Punto':[], 'Iteracion':[], 'AntenaA':[], 'AntenaB':[], 'AntenaC':[], 'AntenaD':[]}

# Leer todos los archivos de texto
for i in range(1, 22): #Recorrer los archivos de P1 a P21
    file_name = f'data_P{i}.txt'
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Contadores
        iteracion = 0 #Para contar las iteraciones de cada archivo
        antena_A = antena_B = antena_C = antena_D = None #Para almacenar los valores RSSI

        for line in lines: 
            # Cuando encontremos una nueva iteracion de redes (cada vez que un scan termina y empieza otro)
            if "Scan done" in line:
                iteracion += 1

                # Cuando ya se hayan leído 20 iteraciones, salir del bucle for
                if iteracion > 20:
                    break
            
            # Extraer valores de RSSI de las antenas correspondientes
            if 'AntenaA' in line:
                antena_A = int(line.split('|')[2].strip())
            elif 'AntenaB' in line:
                antena_B = int(line.split('|')[2].strip())
            elif 'AntenaC' in line:
                antena_C = int(line.split('|')[2].strip())
            elif 'AntenaD' in line:
                antena_D = int(line.split('|')[2].strip())

            #Al final de cada iteracion, agregar los datos al diccionario si se encontraron valores
            if 'Scan start' in line and iteracion<= 20:
                data['Punto'].append(f'P{i}')
                data['Iteracion'].append(iteracion)
                data['AntenaA'].append(antena_A)
                data['AntenaB'].append(antena_B)
                data['AntenaC'].append(antena_C)
                data['AntenaD'].append(antena_D)
                # Reiniciar los valores de las Antenas para la siguiente iteracion
                antena_A = antena_B = antena_C = antena_D = None

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo .txt
output_path = os.path.join(folder_path, 'todosPuntos.txt')
df.to_csv(output_path, sep='\t', index=False)

print (f"Datos guardados en {output_path}")