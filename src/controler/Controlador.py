import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
import os

# Añadir el directorio src a sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.Modelo import BatAlgorithm
from view.Vista import BatView



df = pd.read_csv('C:\\Users\\samue\\OneDrive\\Desktop\\PROYECTO\\data\\CrimesOnWomenData.csv')
print(df.head())  # Primeras filas
print(df.info())  # Información sobre tipos de datos
print(df.describe())  # Estadísticas generales
# Eliminar la columna 'Unnamed: 0'
df = df.drop(columns=['Unnamed: 0'])

# Definir una función para reemplazar outliers
def replace_outliers_with_mean(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definir límites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Reemplazar outliers por la media
        mean_value = df[col].mean()
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), mean_value, df[col])

# Seleccionar columnas numéricas relevantes
numeric_cols = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']

# Aplicar la función a las columnas seleccionadas
replace_outliers_with_mean(df, numeric_cols)

# Mostrar el DataFrame modificado
print(df.head())

# Histogramas
df.hist(bins=30, figsize=(15, 10))
plt.show()

# Boxplots para detectar outliers
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']])
plt.xticks(rotation=45)
plt.show()

# Tablas de frecuencia para variables categóricas
print(df['State'].value_counts())

# Seleccionar solo columnas numéricas
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Calcular la matriz de correlación
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)


# Heatmap para visualizar la correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()



class BatController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self, data_shape):
        self.view.display_initialization(data_shape)
        self.model.initialize(data_shape)
        best_solution = self.model.optimize()
        self.view.display_solution(best_solution)

# Supongamos que 'df' es tu DataFrame y quieres excluir 'State' y 'Year'
data_shape = df.shape[1] - 2

# Crear instancias del modelo, vista y controlador
bat_model = BatAlgorithm()
bat_view = BatView()
bat_controller = BatController(bat_model, bat_view)

# Ejecutar el algoritmo
bat_controller.run(data_shape)
