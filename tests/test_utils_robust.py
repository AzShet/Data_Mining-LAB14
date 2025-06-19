import pytest
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import utils_robust

def test_load_and_engineer_features():
    """
    Prueba que la carga de datos y la ingeniería de características funcionen.
    """
    df = utils_robust.load_and_engineer_features()

    # 1. Comprobar que es un DataFrame de pandas
    assert isinstance(df, pd.DataFrame)

    # 2. Comprobar que 'date' fue eliminada y las nuevas columnas de tiempo existen
    assert 'date' not in df.columns
    assert 'month' in df.columns
    assert 'week_of_year' in df.columns
    assert 'day_of_week' in df.columns

    # 3. Comprobar que las nuevas características de ratio existen
    assert 'incentive_per_target' in df.columns
    assert 'smv_per_worker' in df.columns

    # 4. Comprobar que no hay valores nulos
    assert df.isnull().sum().sum() == 0
