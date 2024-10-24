import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración del puerto serial (ajusta segun el sistema)
ser = serial.Serial('COM4', baudrate=115200, timeout=1) #'COM3' es el puerto serial
time.sleep(2)

#--------------------------------------------------------------------------------------
# Funcion para recolectar los datos de una antena
# recolecta num_puntos desde el puerto serial y los guarda en 'filename'
def recolectar_datos(filename, num_puntos = 600):
    print(f"Recolectando {num_puntos} puntos y guardando en {filename}...")
    with open(filename, 'w') as file:
        for i in range (num_puntos):
            data = ser.readline().decode('utf-8').strip() #lee y decodifica los datos ascii
            if data: 
                file.write(f'{data}\n') #guardar en el archivo .txt
                print(f'Dato {i+1}: {data}')
            time.sleep(0.1) #pausa entre las lecturas
    print(f"Datos guardados en {filename}")

# Recolectar los datos de cada antena (manda a llamar la funcion)
# Cambiar el nombre del archivo segun la antena que se este recolectando
nombre_prueba = 'P21' # esto se cambia segun la antena
recolectar_datos(f'data_{nombre_prueba}.txt')

# Cerrar la conexion serial
ser.close()