import pytest
import pandas as pd
import numpy as np
import sys
import os

# Añadir el directorio raíz del proyecto al path para encontrar el módulo src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import utils

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture que crea un DataFrame de pandas de ejemplo para las pruebas."""
    data = {
        'num_col1': [1, 2, 3, 4, 100],
        'num_col2': [10.0, 20.0, np.nan, 40.0, 50.0],
        'cat_col': ['A', 'B', 'A', 'C', 'B'],
        'target': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return pd.DataFrame(data)

def test_load_and_prepare_data():
    """Prueba que la carga de datos funciona y elimina la columna 'date'."""
    df = utils.load_and_prepare_data()
    assert isinstance(df, pd.DataFrame)
    assert 'date' not in df.columns
    assert 'actual_productivity' in df.columns

def test_handle_missing_values(sample_df):
    """Prueba que los valores nulos son imputados correctamente."""
    df_imputed = utils.handle_missing_values(sample_df)
    assert df_imputed['num_col2'].isnull().sum() == 0
    # La mediana de [10, 20, 40, 50] es 30.0
    assert df_imputed.loc[2, 'num_col2'] == 30.0

def test_convert_to_dummies(sample_df):
    """Prueba la conversión a variables dummy."""
    df_dummies = utils.convert_to_dummies(sample_df, cat_cols=['cat_col'])
    assert 'cat_col' not in df_dummies.columns
    assert 'cat_col_B' in df_dummies.columns
    assert 'cat_col_C' in df_dummies.columns
    assert 'cat_col_A' not in df_dummies.columns

def test_treat_outliers_iqr(sample_df):
    """Prueba que los outliers son tratados (capped)."""
    num_cols = ['num_col1']
    df_treated = utils.treat_outliers_iqr(sample_df.copy(), num_cols)
    Q1 = sample_df['num_col1'].quantile(0.25) # 2.5
    Q3 = sample_df['num_col1'].quantile(0.75) # 4.5 -> This is incorrect for [1,2,3,4,100] pandas quantile is different. Let's calculate manually.
    # Sorted: [1, 2, 3, 4, 100]. Q1=2.0, Q3=4.0, IQR=2.0. Upper bound = 4.0 + 1.5*2.0 = 7.0
    upper_bound = 4.0 + 1.5 * (4.0 - 2.0)
    assert df_treated.loc[4, 'num_col1'] == upper_bound
