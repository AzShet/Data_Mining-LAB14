import logging
import pyarrow
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_prepare_data() -> pd.DataFrame:
    """
    Carga el dataset de UCI, lo combina y lo retorna como un DataFrame de pandas.
    Usa Polars internamente para una operación de limpieza rápida.

    Returns:
        pd.DataFrame: El DataFrame de pandas listo para el preprocesamiento.
    """
    logging.info("Iniciando la carga de datos desde el repositorio UCI...")
    try:
        garment_prod = fetch_ucirepo(id=597)
        X_pd = garment_prod.data.features
        y_pd = garment_prod.data.targets

        df_pd = pd.concat([X_pd, y_pd], axis=1)

        # --- Uso de Polars para una operación eficiente ---
        # Convertimos a Polars para eliminar la columna de forma rápida y segura
        df_pl = pl.from_pandas(df_pd)
        if 'date' in df_pl.columns:
            df_pl = df_pl.drop('date')
            logging.info("Columna 'date' eliminada usando Polars.")

        # Regresamos a pandas para continuar el flujo de trabajo
        df_final_pd = df_pl.to_pandas()
        # -------------------------------------------------

        logging.info(f"Datos cargados exitosamente. Dimensiones del DataFrame: {df_final_pd.shape}")
        return df_final_pd
    except Exception as e:
        logging.error(f"Error al cargar o preparar los datos: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa los valores faltantes en un DataFrame de pandas usando la mediana.

    Args:
        df (pd.DataFrame): El DataFrame con posibles valores nulos.

    Returns:
        pd.DataFrame: El DataFrame sin valores nulos.
    """
    logging.info("Iniciando tratamiento de datos faltantes.")
    missing_counts = df.isnull().sum().sum()

    if missing_counts == 0:
        logging.info("No se encontraron valores faltantes.")
        return df

    logging.info(f"Total de valores faltantes encontrados: {missing_counts}")

    # Iterar sobre columnas con nulos y rellenar con la mediana
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"Valores nulos en '{col}' imputados con la mediana ({median_val}).")

    return df

def treat_outliers_iqr(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Trata los outliers en columnas numéricas usando el método de capping por IQR.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        num_cols (List[str]): Lista de columnas numéricas a tratar.

    Returns:
        pd.DataFrame: DataFrame con outliers tratados.
    """
    logging.info("Iniciando tratamiento de outliers a nivel univariado (IQR).")
    df_out = df.copy()

    for col in num_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()

        if outliers_count > 0:
            logging.info(f"Tratando {outliers_count} outliers en la columna '{col}'.")
            df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

    logging.info("Tratamiento de outliers finalizado.")
    return df_out

def convert_to_dummies(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Convierte las columnas categóricas a variables dummy usando pandas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        cat_cols (List[str]): Lista de columnas categóricas.

    Returns:
        pd.DataFrame: DataFrame con variables dummy.
    """
    logging.info("Convirtiendo variables categóricas a dummies...")
    if not cat_cols:
        logging.info("No hay columnas categóricas para convertir.")
        return df

    df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    logging.info(f"Columnas convertidas a dummies: {cat_cols}")
    return df_dummies

def train_and_evaluate_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    random_state: int
) -> pd.DataFrame:
    """
    Entrena y evalúa varios modelos de ensamblaje.

    Args:
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (pd.Series): Objetivo de entrenamiento.
        X_test (pd.DataFrame): Características de prueba.
        y_test (pd.Series): Objetivo de prueba.
        random_state (int): Semilla aleatoria para reproducibilidad.

    Returns:
        pd.DataFrame: Un DataFrame con las métricas de evaluación para cada modelo.
    """
    logging.info("Iniciando entrenamiento y evaluación de modelos de ensamblaje.")

    # --- Definición de modelos base con hiperparámetros ---
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state)
    r3 = KNeighborsRegressor(n_neighbors=10)
    r4 = SVR(kernel='rbf', C=1.0)
    r5 = DecisionTreeRegressor(max_depth=5, random_state=random_state)

    # --- Modelos de Ensamblaje ---
    models = {
        'Voting': VotingRegressor(estimators=[('lr', r1), ('rf', r2), ('knn', r3)]),
        'Bagging': BaggingRegressor(estimator=r5, n_estimators=50, random_state=random_state),
        'Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
        'Stacking': StackingRegressor(estimators=[('rf', r2), ('knn', r3), ('svr', r4)], final_estimator=r1)
    }

    results = []
    for name, model in models.items():
        logging.info(f"Entrenando el modelo: {name}...")
        model.fit(X_train, y_train)

        logging.info(f"Evaluando el modelo: {name}...")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results.append({
            "Modelo": name,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        })
    logging.info("Evaluación de todos los modelos completada.")
    return pd.DataFrame(results)
