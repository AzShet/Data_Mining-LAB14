import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
# --- IMPORTACIÓN AÑADIDA ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# Obtenemos el logger para este módulo
logger = logging.getLogger(__name__)

def load_and_engineer_features() -> pd.DataFrame:
    """
    Carga los datos y realiza ingeniería de características avanzada.
    """
    logger.info("Iniciando carga de datos e ingeniería de características...")
    try:
        garment_prod = fetch_ucirepo(id=597)
        df = pd.concat([garment_prod.data.features, garment_prod.data.targets], axis=1)

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        df = df.drop(columns=['date'])
        logger.info("Características de tiempo creadas desde la columna 'date'.")

        df['incentive_per_target'] = df['incentive'] / (df['targeted_productivity'] + 1e-6)
        df['smv_per_worker'] = df['smv'] / (df['no_of_workers'] + 1e-6)
        logger.info("Características de ratio creadas.")

        for col in df.columns[df.isnull().any()]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        logger.info("Valores nulos imputados con la mediana.")

        logger.info(f"Proceso finalizado. Dimensiones del DataFrame: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error durante la carga e ingeniería de características: {e}")
        raise

def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int, cv: int, random_state: int) -> Tuple[object, Dict]:
    """
    Busca el mejor modelo y sus hiperparámetros usando RandomizedSearchCV.
    """
    logger.info("Iniciando la búsqueda del mejor modelo robusto...")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- CAMBIO EN EL PREPROCESADOR ---
    # Se reemplaza 'passthrough' por OneHotEncoder para las variables categóricas.
    # handle_unknown='ignore' evita errores durante la validación cruzada si una categoría
    # no está presente en algún pliegue (fold).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    search_spaces = {
        'RandomForest': (
            RandomForestRegressor(random_state=random_state),
            {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, 30],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2'],
            }
        ),
        'XGBoost': (
            XGBRegressor(random_state=random_state, objective='reg:squarederror'),
            {
                'model__n_estimators': [100, 200, 500],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.7, 0.8, 0.9],
                'model__colsample_bytree': [0.7, 0.8, 0.9],
            }
        ),
        'LightGBM': (
            LGBMRegressor(random_state=random_state),
            {
                'model__n_estimators': [100, 200, 500],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__num_leaves': [20, 31, 40],
                'model__max_depth': [-1, 10, 20],
                'model__subsample': [0.7, 0.8, 0.9],
            }
        )
    }

    best_model = None
    best_score = -np.inf
    results = {}

    for name, (model, params) in search_spaces.items():
        logger.info(f"--- Optimizando {name} ---")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        search = RandomizedSearchCV(
            pipeline, params, n_iter=n_iter, scoring='r2', n_jobs=-1,
            cv=cv, verbose=1, random_state=random_state
        )
        search.fit(X_train, y_train)

        logger.info(f"Mejor R2 score para {name} (CV): {search.best_score_:.4f}")
        logger.info(f"Mejores parámetros: {search.best_params_}")

        results[name] = {'best_score': search.best_score_, 'best_params': search.best_params_}

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            results['best_overall_model'] = name

    logger.info(f"Búsqueda finalizada. Mejor modelo global: {results['best_overall_model']}")
    return best_model, results
