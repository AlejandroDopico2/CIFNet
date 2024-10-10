import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def cargar_jsons(archivos_json):
    resultados = []
    for archivo in archivos_json:
        with open(archivo, 'r') as f:
            resultados.append(json.load(f))
    return resultados

def obtener_nombre_carpeta(archivo_json):
    # Obtiene el nombre de la carpeta donde está almacenado el archivo JSON
    carpeta = os.path.basename(os.path.dirname(archivo_json))
    return carpeta

def calcular_promedio_resultados(resultados):
    promedios = []
    for resultado in resultados:
        # Calcular el promedio de los últimos valores de todas las tareas
        valores = [resultado[tarea][-1] for tarea in resultado.keys()]
        promedio = sum(valores) / len(valores)
        promedios.append(promedio)
    return promedios

def graficar_resultados(resultados, nombres_carpetas):
    tareas = list(resultados[0].keys())  # Asume que todas tienen las mismas tareas
    num_resultados = len(resultados)

    # Crear la gráfica
    x = np.arange(len(tareas) + 1)  # la ubicación de las etiquetas, +1 para los promedios
    width = 0.8 / num_resultados  # el ancho de las barras, ajustado al número de resultados

    fig, ax = plt.subplots()

    # Calcular y graficar las barras de precisión media (promedio por cada modelo)
    promedios = calcular_promedio_resultados(resultados)

    # Graficar resultados de las tareas y agregar una barra para el promedio
    for i, (resultado, nombre_carpeta) in enumerate(zip(resultados, nombres_carpetas)):
        valores_tareas = [resultado[tarea][-1] for tarea in tareas]  # Toma el último valor de cada tarea
        valores_con_promedio = valores_tareas + [promedios[i]]  # Añadir el promedio al final de las tareas

        bars = ax.bar(x - width * (num_resultados / 2 - i), valores_con_promedio, width, label=nombre_carpeta)

        # Añadir los valores sobre las barras
        for barra in bars:
            altura = barra.get_height()
            ax.annotate(
                f"{altura:.3f}",
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 3),  # 3 puntos de desplazamiento hacia arriba
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Ajustar etiquetas y leyenda
    ax.set_ylabel("Valores")
    ax.set_title("Comparación de Resultados")
    ax.set_xticks(x)
    ax.set_xticklabels(tareas + ["Promedio"])  # Etiquetas de las tareas + promedio
    ax.legend()

    plt.tight_layout()
    plt.savefig(".")

def main():
    parser = argparse.ArgumentParser(description="Comparar resultados de múltiples archivos JSON")
    parser.add_argument('json_files', nargs='+', help='Archivos JSON a comparar')
    args = parser.parse_args()

    # Cargar los JSONs
    resultados = cargar_jsons(args.json_files)

    # Obtener los nombres de las carpetas para la leyenda
    nombres_carpetas = [obtener_nombre_carpeta(json_file) for json_file in args.json_files]

    # Graficar los resultados
    graficar_resultados(resultados, nombres_carpetas)

if __name__ == "__main__":
    main()
