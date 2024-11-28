import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score

# Añadir el directorio src a sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.Modelo import BatAlgorithm
from view.Vista import BatView

class BatController:
    def __init__(self, model, view, data_path, data_shape=None):
        self.model = model
        self.view = view
        self.data_path = data_path
        self.data = self.load_data(data_path)
        self.data_shape = data_shape

    def load_data(self, data_path):
        """
        Carga los datos desde un archivo CSV y realiza limpieza inicial.
        """
        df = pd.read_csv(data_path)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Eliminar columnas innecesarias
        return df

    def replace_outliers(self, df, cols):
        """
        Reemplaza valores atípicos en las columnas seleccionadas.
        """
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mean_value = df[col].mean()
            df[col] = np.where(
                (df[col] < lower_bound) | (df[col] > upper_bound), mean_value, df[col]
            )

    def normalize_data(self, df, cols):
        """
        Normaliza las columnas seleccionadas con MinMaxScaler.
        """
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])

    def plot_boxplot(self, df, cols):
        """
        Genera un boxplot de las columnas seleccionadas.
        """
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df[cols])
        plt.xticks(rotation=45)
        plt.title("Boxplot de Datos")
        plt.show()

    def plot_correlation_heatmap(self, df):
        """
        Genera un heatmap para visualizar la correlación entre columnas numéricas.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Mapa de Calor de Correlación")
        plt.show()

    def evaluate_model(self, X, y, best_solution):
        """
        Evalúa el modelo utilizando las métricas de rendimiento.
        """
        # Predicciones utilizando la mejor solución obtenida
        y_pred = X.dot(best_solution)  # Asumimos que la predicción es una combinación lineal

        # Calcular el MSE y el R2
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Puedes agregar más métricas según sea necesario
        return mse, r2

    def run(self):
        """
        Ejecuta el flujo completo: carga, preprocesamiento, visualización, optimización y evaluación.
        """
        print("Cargando y procesando datos...")
        
        # Cargar y procesar datos
        df = self.data
        numeric_cols = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
        self.replace_outliers(df, numeric_cols)
        self.normalize_data(df, numeric_cols)

        # Visualizar datos procesados
        self.plot_boxplot(df, numeric_cols)
        self.plot_correlation_heatmap(df)

        # Separar las variables predictoras y de salida
        X = df.drop(columns=['State', 'Year', 'Rape'])  # Eliminar columnas no necesarias
        y = df['Rape']  # 'Rape' como variable de salida (target)
        
        # Inicializar el modelo de optimización
        data_shape = X.shape[1]  # Número de características
        self.model.initialize(data_shape)
        
        # Ejecutar el modelo
        best_solution = self.model.optimize()  # Llamar a optimize() sin pasar X, y

        # Evaluar el modelo
        mse, r2 = self.evaluate_model(X, y, best_solution)

        # Visualización de la solución
        self.view.display_solution(best_solution)

        return mse, r2  # Retorna las métricas de evaluación para el seguimiento



from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BatPipeline:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path

    def load_and_preprocess_data(self):
        """
        Carga y preprocesa los datos.
        """
        df = pd.read_csv(self.data_path)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Eliminar columnas innecesarias
        numeric_cols = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']

        # Reemplazar outliers
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mean_value = df[col].mean()
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), mean_value, df[col])

        # Normalizar datos
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df

    def create_pipeline(self):
        """
        Define y crea el pipeline de procesamiento de datos.
        """
        df = self.load_and_preprocess_data()

        # Separar las variables predictoras y de salida (target)
        X = df.drop(columns=['State', 'Year', 'Rape'])  # Eliminar columnas no numéricas
        y = df['Rape']  # Supongo que 'Rape' es la variable de interés (target)
        
        # Transformar los datos con el pipeline
        return X, y

    def run(self):
        """
        Ejecuta el flujo completo: carga, preprocesamiento, visualización y optimización.
        """
        print("Cargando y procesando datos...")

        # Cargar y procesar los datos con el pipeline
        df = self.load_and_preprocess_data()
        numeric_cols = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
        
        # Visualizar datos procesados (Boxplot y Heatmap)
        self.plot_boxplot(df, numeric_cols)
        self.plot_correlation_heatmap(df)

        # Separar las variables predictoras y de salida
        X = df.drop(columns=['State', 'Year', 'Rape'])  # Eliminar columnas no necesarias
        y = df['Rape']  # 'Rape' como variable de salida (target)
        
        # Inicializar el modelo de optimización
        data_shape = X.shape[1]  # Número de características
        self.model.initialize(data_shape)
        
        # Ejecutar el modelo de optimización
        best_solution = self.model.optimize()  # Ejecutar sin pasar X, y, ya que el modelo los maneja internamente

        # Mostrar la solución optimizada
        print("Mejor solución encontrada:", best_solution)

        # Visualizar la solución con la vista
        self.view.display_solution(best_solution)

    def plot_boxplot(self, df, cols):
        """
        Genera un boxplot de las columnas seleccionadas.
        """
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df[cols])
        plt.xticks(rotation=45)
        plt.title("Boxplot de Datos")
        plt.show()

    def plot_correlation_heatmap(self, df):
        """
        Genera un heatmap para visualizar la correlación entre columnas numéricas.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Mapa de Calor de Correlación")
        plt.show()

# Instanciar el modelo y la vista
bat_model = BatAlgorithm()  # Instancia del modelo Bat
bat_view = BatView()  # Instancia de la vista
data_path = 'C:\\Users\\samue\\OneDrive\\Desktop\\PROYECTO\\data\\CrimesOnWomenData.csv'  # Ruta a los datos

# Crear instancia del controlador y ejecutarlo
bat_controller = BatController(bat_model, bat_view, data_path)
bat_controller.run()
