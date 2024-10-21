import matplotlib.pyplot as plt
import numpy as np

class BatView:
    def display_solution(self, solution):
        print(f"Mejor solución encontrada: {solution}")
        self.plot_solution(solution)

    def display_initialization(self, data_shape):
        print(f"Inicializando con {data_shape} dimensiones.")

    def plot_solution(self, solution):
        # Nombres para las dimensiones (puedes personalizarlos)
        dimensions = [f'Dimensión {i+1}' for i in range(len(solution))]

        # Crear gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.bar(dimensions, solution, color='skyblue')
        plt.xlabel('Dimensiones')
        plt.ylabel('Valores de la Mejor Solución')
        plt.title('Mejor Solución Encontrada por el Algoritmo de Murciélago')
        plt.axhline(0, color='red', linewidth=0.8, linestyle='--')  # Línea horizontal en y=0
        plt.xticks(rotation=45)
        plt.grid(axis='y')

        # Mostrar gráfico
        plt.tight_layout()
        plt.show()
