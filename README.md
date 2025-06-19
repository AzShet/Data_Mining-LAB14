# Productividad de Empleados de la Confección - Análisis y Predicción

Este proyecto utiliza el dataset "Productivity Prediction of Garment Employees" de UCI para construir y evaluar modelos de machine learning que predicen la productividad real de los empleados.

## Estructura del Repositorio

- **/notebooks**: Contiene el Jupyter Notebook `LAB14-RUELAS.ipynb` con el flujo completo de análisis, preprocesamiento, modelado y evaluación.
- **/src**: Contiene el módulo de Python `utils.py` con todas las funciones auxiliares utilizadas en el notebook.
- **/tests**: Contiene las pruebas unitarias para las funciones en `src/utils.py` utilizando `pytest`.
- **requirements.txt**: Lista de las dependencias de Python necesarias para ejecutar el proyecto.
- **.gitignore**: Archivo para excluir archivos y directorios irrelevantes del control de versiones.

## Objetivo

El objetivo principal es predecir la variable `actual_productivity` utilizando diversas técnicas de ensamblaje de modelos (Voting, Bagging, Boosting, Stacking).

## Flujo de Trabajo

1.  **Carga y Limpieza de Datos**: Se carga el dataset usando la librería `ucimlrepo` y se convierte a un DataFrame de Polars. Se eliminan columnas innecesarias.
2.  **Preprocesamiento**:
    -   Tratamiento de valores faltantes.
    -   Manejo de outliers a nivel univariado.
    -   Conversión de variables categóricas a formato dummy.
3.  **Modelado**:
    -   División de los datos en conjuntos de entrenamiento (80%) y prueba (20%).
    -   Escalado de las variables numéricas (StandardScaler).
    -   Entrenamiento de modelos de ensamblaje basados en k-NN, SVM, Regresión Lineal, Árbol de Decisión y Random Forest.
4.  **Evaluación**:
    -   Se utilizan las métricas MSE, RMSE, MAE y R² para comparar el rendimiento de los modelos.
    -   Se selecciona y justifica el mejor modelo basado en los resultados.

## Cómo Ejecutar

1.  Clona el repositorio.
2.  Crea un entorno virtual: `python -m venv venv`
3.  Activa el entorno: `source venv/bin/activate` (en Linux/macOS) o `venv\Scripts\activate` (en Windows).
4.  Instala las dependencias: `pip install -r requirements.txt`
5.  Navega a la carpeta `notebooks` y abre el notebook: `jupyter lab LAB14-RUELAS.ipynb`
6.  Ejecuta las celdas del notebook en orden. Las pruebas se pueden ejecutar con el comando `!pytest ../ -v` desde una celda del notebook.