"""Organized workflow for downloading, cleaning, exploring and modeling the
pensiones dataset from datos.gov.co.

This module preserves the original analysis logic from ``pensiones.py`` but
reorders it into a coherent pipeline so each phase (ingest → limpieza →
subsets → EDA → modelado) can be executed in sequence.
"""
from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import entropy as shannon_entropy
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------------------------------------------------------
# Rutas básicas
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELOS_DIR = DATA_DIR / "modelos"
GRAFICAS_COMPARATIVAS_DIR = DATA_DIR / "graficas_comparativas"
GRAFICAS_MODELADO_DIR = DATA_DIR / "graficas_modelado"
EDA_IMAGES_DIR = BASE_DIR / "images" / "analisis_exploratorio"

BASE = "https://www.datos.gov.co"
RESOURCE = "uawh-cjvi"
URL = f"{BASE}/resource/{RESOURCE}.json"
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def slugify(value: str) -> str:
    """Convierte cadenas a slug amigable para nombres de archivo."""
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
        "Á": "a",
        "É": "e",
        "Í": "i",
        "Ó": "o",
        "Ú": "u",
        "Ñ": "n",
        " ": "_",
        "-": "-",
        "\"": "",
        ".": ""
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.lower()


def ensure_directories():
    for path in [RAW_DIR, PROCESSED_DIR, MODELOS_DIR,
                 GRAFICAS_COMPARATIVAS_DIR, GRAFICAS_MODELADO_DIR, EDA_IMAGES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def guardar_subset(df: pd.DataFrame, filtro: str, valores, salida: str) -> pd.DataFrame:
    if isinstance(valores, (list, tuple, set)):
        df_subset = df.loc[df[filtro].isin(valores)].copy()
    else:
        df_subset = df.loc[df[filtro].eq(valores)].copy()
    if filtro in df_subset.columns:
        df_subset = df_subset.drop(columns=[filtro])

    print(df_subset.shape)
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(salida, index=False)
    return df_subset


COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown']


def graficar_comparacion_entidades_por_fondo(
    fondos_a_comparar: Dict[str, pd.DataFrame],
    titulo_base: str,
    nombre_archivo: str,
) -> None:
    plt.figure(figsize=(14, 8))
    for i, (entidad, df_fondo) in enumerate(fondos_a_comparar.items()):
        if len(df_fondo) > 0:
            color = COLORS[i % len(COLORS)]
            plt.plot(
                df_fondo['fecha'],
                df_fondo['valor_unidad'],
                label=entidad,
                color=color,
                linewidth=2,
                alpha=0.8,
            )
    plt.title(f'{titulo_base} - Comparación por Entidad', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor Unidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(GRAFICAS_COMPARATIVAS_DIR / f"{nombre_archivo}.png", dpi=300, bbox_inches='tight')
    plt.close()


def evolucion_todos_fondos_entidad(entidad_nombre: str, dataframes_fondos: Dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(16, 10))
    fondos_colores = {
        'Cesantías Largo Plazo': 'blue',
        'Cesantías Corto Plazo': 'lightblue',
        'Pensiones Moderado': 'green',
        'Pensiones Conservador': 'darkgreen',
        'Pensiones Mayor Riesgo': 'red',
        'Pensiones Retiro Programado': 'orange',
        'Pensiones Alternativo': 'purple',
    }
    for fondo_nombre, color in fondos_colores.items():
        if fondo_nombre in dataframes_fondos and len(dataframes_fondos[fondo_nombre]) > 0:
            df_fondo = dataframes_fondos[fondo_nombre]
            valor_base = df_fondo['valor_unidad'].iloc[0]
            df_normalizado = (df_fondo['valor_unidad'] / valor_base * 100)
            plt.plot(
                df_fondo['fecha'],
                df_normalizado,
                label=fondo_nombre,
                color=color,
                linewidth=2,
                alpha=0.7,
            )
    plt.title(
        f'Evolución de Todos los Fondos - {entidad_nombre}\n(Valor de la Unidad Normalizado Base 100)',
        fontsize=14,
        fontweight='bold',
    )
    plt.xlabel('Fecha')
    plt.ylabel('Valor Normalizado (Base 100)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        GRAFICAS_COMPARATIVAS_DIR / f'evolucion_todos_fondos_{slugify(entidad_nombre)}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def matriz_correlacion_fondos(entidad_nombre: str, dataframes_fondos: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    datos_correlacion = {}
    for fondo_nombre, df in dataframes_fondos.items():
        if len(df) > 0:
            df_temp = df.set_index('fecha')['valor_unidad'].sort_index()
            returns = df_temp.pct_change().dropna()
            datos_correlacion[fondo_nombre] = returns
    df_correlacion = pd.DataFrame(datos_correlacion)
    matriz_corr = df_correlacion.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
    sns.heatmap(
        matriz_corr,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Coeficiente de Correlación'},
    )
    plt.title(f'Matriz de Correlación - {entidad_nombre}\n(Returns Diarios)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        GRAFICAS_COMPARATIVAS_DIR / f'correlacion_{slugify(entidad_nombre)}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    return matriz_corr

# -----------------------------------------------------------------------------
# Funciones de modelado temporal (idénticas al script original)
# -----------------------------------------------------------------------------

def analizar_estacionariedad(serie: pd.Series, nombre_serie: str = "") -> Dict[str, object]:
    print(f"\n--- Análisis de Estacionariedad: {nombre_serie} ---")
    resultado_adf = adfuller(serie.dropna())
    metricas = {
        'estadistico_adf': resultado_adf[0],
        'p_valor': resultado_adf[1],
        'valores_criticos': resultado_adf[4],
        'es_estacionaria': resultado_adf[1] < 0.05,
    }
    print(f"Estadístico ADF: {metricas['estadistico_adf']:.4f}")
    print(f"P-valor: {metricas['p_valor']:.4f}")
    if metricas['es_estacionaria']:
        print("✓ La serie ES estacionaria (p-valor < 0.05)")
    else:
        print("✗ La serie NO es estacionaria (p-valor > 0.05)")
        print("  Se requiere diferenciación para modelado ARIMA")
    return metricas


def entrenar_modelo_arima(
    serie: pd.Series,
    orden: Tuple[int, int, int],
    nombre_serie: str = "",
):
    print(f"Entrenando ARIMA{orden} para {nombre_serie}...")
    try:
        modelo = ARIMA(serie, order=orden)
        modelo_ajustado = modelo.fit()
        metricas = {
            'aic': modelo_ajustado.aic,
            'bic': modelo_ajustado.bic,
            'residuos_media': modelo_ajustado.resid.mean(),
            'residuos_std': modelo_ajustado.resid.std(),
        }
        print(f"✓ ARIMA{orden} entrenado exitosamente")
        print(f"  AIC: {metricas['aic']:.2f}, BIC: {metricas['bic']:.2f}")
        return modelo_ajustado, metricas
    except Exception as exc:  # pragma: no cover (logs)
        print(f"✗ Error entrenando ARIMA{orden}: {exc}")
        return None, None


def buscar_mejor_arima(
    serie: pd.Series,
    parametros_a_probar: List[Tuple[int, int, int]],
    nombre_serie: str = "",
):
    print(f"\nBuscando mejor modelo ARIMA para {nombre_serie}...")
    mejores_metricas = {'aic': float('inf')}
    mejor_modelo = None
    mejor_orden = None
    for orden in parametros_a_probar:
        modelo, metricas = entrenar_modelo_arima(serie, orden, nombre_serie)
        if modelo and metricas and metricas['aic'] < mejores_metricas['aic']:
            mejores_metricas = metricas
            mejor_modelo = modelo
            mejor_orden = orden
    if mejor_modelo:
        print("\n🎯 MEJOR MODELO ENCONTRADO:")
        print(f"   ARIMA{mejor_orden} con AIC: {mejores_metricas['aic']:.2f}")
        return {'modelo': mejor_modelo, 'orden': mejor_orden, 'metricas': mejores_metricas}
    print("✗ No se pudo encontrar un modelo adecuado")
    return None


def evaluar_pronostico(real, pronosticado, nombre_serie=""):
    print(f"\n--- Evaluación de Pronósticos: {nombre_serie} ---")
    mse = np.mean((real - pronosticado) ** 2)
    mae = np.mean(np.abs(real - pronosticado))
    mape = np.mean(np.abs((real - pronosticado) / real)) * 100
    rmse = np.sqrt(mse)
    metricas = {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
    print("Métricas de evaluación:")
    for metrica, valor in metricas.items():
        print(f"  {metrica}: {valor:.4f}")
    print(f"  Error porcentual promedio (MAPE): {mape:.2f}%")
    return metricas


def pipeline_modelado_completo(df_serie, nombre_serie, columna_valor='valor_unidad'):
    print(f"\n{'=' * 50}")
    print(f"INICIANDO PIPELINE DE MODELADO: {nombre_serie}")
    print(f"{'=' * 50}")
    resultados = {}
    serie = df_serie.set_index('fecha')[columna_valor].sort_index()
    resultados['serie_original'] = serie.copy()
    print("\n2. 🔍 ANALIZANDO ESTACIONARIEDAD...")
    resultados['estacionariedad'] = analizar_estacionariedad(serie, nombre_serie)
    if not resultados['estacionariedad']['es_estacionaria']:
        print("\n3. ⚙️  APLICANDO DIFERENCIACIÓN...")
        serie_diff = serie.diff().dropna()
        resultados['serie_diferenciada'] = serie_diff
        resultados['estacionariedad_diff'] = analizar_estacionariedad(
            serie_diff, f"{nombre_serie} (diferenciada)"
        )
        serie_para_modelar = serie_diff
        resultados['usa_diferencia'] = True
    else:
        serie_para_modelar = serie
        resultados['usa_diferencia'] = False
    print("\n4. 🤖 BUSCANDO MEJOR MODELO ARIMA...")
    if resultados['usa_diferencia']:
        parametros_a_probar = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)]
    else:
        parametros_a_probar = [(1, 0, 0), (1, 1, 1), (2, 1, 2), (0, 1, 1), (1, 1, 0)]
    resultados['serie_para_modelar'] = serie_para_modelar
    resultados['mejor_modelo'] = buscar_mejor_arima(
        serie_para_modelar, parametros_a_probar, nombre_serie
    )
    if resultados['mejor_modelo']:
        modelo = resultados['mejor_modelo']['modelo']
        print("\n5. ✅ VALIDANDO MODELO...")
        train_size = int(len(serie_para_modelar) * 0.8)
        train, test = serie_para_modelar[:train_size], serie_para_modelar[train_size:]
        modelo_train = ARIMA(train, order=resultados['mejor_modelo']['orden']).fit()
        pronostico = modelo_train.forecast(steps=len(test))
        resultados['evaluacion'] = evaluar_pronostico(
            test.values, pronostico.values, nombre_serie
        )
        print("\n6. 🔮 GENERANDO PRONÓSTICOS FUTUROS...")
        resultados['pronostico_futuro'] = modelo.forecast(steps=30)
        print("📈 Pronóstico para los próximos 30 pasos del modelo (no aún en niveles).")
    return resultados

def visualizar_resultados_modelado(resultados_modelado):
    print("\n🎨 GENERANDO VISUALIZACIONES DE RESULTADOS...")
    GRAFICAS_MODELADO_DIR.mkdir(parents=True, exist_ok=True)
    for nombre_serie, resultados in resultados_modelado.items():
        if not resultados.get('mejor_modelo'):
            continue
        try:
            serie_original = resultados['serie_original']
            modelo = resultados['mejor_modelo']['modelo']
            usa_dif = resultados.get('usa_diferencia', False)
            if usa_dif:
                serie_diff = resultados['serie_diferenciada']
                fitted_diff = modelo.fittedvalues
                fitted_diff = fitted_diff.loc[serie_diff.index]
                first_idx = serie_diff.index[0]
                pos_prev = serie_original.index.get_loc(first_idx) - 1
                base_level = serie_original.iloc[pos_prev]
                fitted_level = base_level + fitted_diff.cumsum()
                pronostico_diff = resultados['pronostico_futuro']
                last_level = serie_original.iloc[-1]
                forecast_level = last_level + pronostico_diff.cumsum()
            else:
                fitted_level = modelo.fittedvalues
                forecast_level = resultados.get('pronostico_futuro')
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(serie_original.index, serie_original.values, label='Original', alpha=0.7)
            idx_comun = serie_original.index.intersection(fitted_level.index)
            plt.plot(idx_comun, fitted_level.loc[idx_comun], label='Ajustado (reconstruido)', alpha=0.9)
            plt.title(f'Serie Original vs Ajustada\n{nombre_serie}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.subplot(2, 2, 2)
            residuos = modelo.resid
            plt.plot(residuos.index, residuos.values)
            plt.title('Residuos del Modelo')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xticks(rotation=45)
            plt.subplot(2, 2, 3)
            plt.hist(residuos.dropna(), bins=50, alpha=0.7, density=True)
            plt.title('Distribución de Residuos')
            plt.xlabel('Residuos')
            plt.ylabel('Densidad')
            plt.subplot(2, 2, 4)
            if forecast_level is not None:
                ultimos_30 = serie_original.tail(30)
                plt.plot(ultimos_30.index, ultimos_30.values, label='Últimos 30 días', color='blue')
                plt.plot(
                    forecast_level.index,
                    forecast_level.values,
                    label='Pronóstico (niveles)',
                    color='red',
                    linestyle='--',
                )
                plt.title('Pronóstico en la escala real')
                plt.legend()
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                GRAFICAS_MODELADO_DIR / f"resultados_{nombre_serie.replace(' ', '_').lower()}.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()
            print(f"✓ Gráficas corregidas guardadas para: {nombre_serie}")
        except Exception as exc:  # pragma: no cover (logs)
            print(f"✗ Error generando gráficas para {nombre_serie}: {exc}")


# -----------------------------------------------------------------------------
# SARIMA utilities from the original script
# -----------------------------------------------------------------------------

def _ensure_monthly_close(df, col_fecha='fecha', col_valor='valor_unidad'):
    s = (
        df[[col_fecha, col_valor]]
        .dropna()
        .sort_values(col_fecha)
        .drop_duplicates(subset=[col_fecha])
        .set_index(col_fecha)[col_valor]
        .astype(float)
        .sort_index()
    )
    s_m = s.resample('M').last().asfreq('M').ffill()
    return s_m


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _grid_search_sarima(y, pdq_grid, seasonal_pdq_grid, trend='c'):
    best = {'aic': np.inf, 'order': None, 'seasonal_order': None, 'model': None}
    for order in pdq_grid:
        for sorder in seasonal_pdq_grid:
            try:
                model = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=sorder,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                if res.aic < best['aic']:
                    best = {'aic': res.aic, 'order': order, 'seasonal_order': sorder, 'model': res}
            except Exception:
                continue
    return best


def _plot_fit_vs_real(y, fit_in, fc, fc_ci, titulo, out_png):
    plt.figure(figsize=(14, 7))
    plt.plot(y.index, y.values, label='Real (valor de la unidad)', alpha=0.8)
    if fit_in is not None:
        plt.plot(fit_in.index, fit_in.values, label='Ajuste in-sample (modelo)', linewidth=2)
    if fc is not None:
        plt.plot(fc.index, fc.values, '--', label='Pronóstico', linewidth=2)
        if fc_ci is not None:
            plt.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], alpha=0.15, label='IC 95%')
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la unidad')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def entrenar_y_graficar_sarima(
    df_serie,
    nombre_serie,
    meses_test=24,
    pasos_forecast=12,
    pdq=((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2)),
    spdq=((0, 0, 0, 12), (1, 0, 0, 12), (0, 1, 1, 12), (1, 1, 1, 12)),
):
    y = _ensure_monthly_close(df_serie, 'fecha', 'valor_unidad')
    if len(y) < (meses_test + 36):
        print(f"⚠️ Serie '{nombre_serie}' es muy corta para validar (len={len(y)}). Se omite.")
        return None
    y_train = y.iloc[:-meses_test]
    y_test = y.iloc[-meses_test:]
    pdq_grid = list(pdq)
    seasonal_pdq_grid = list(spdq)
    best = _grid_search_sarima(y_train, pdq_grid, seasonal_pdq_grid, trend='c')
    if best['model'] is None:
        print(f"✗ No se encontró modelo para '{nombre_serie}'.")
        return None
    order, sorder, res = best['order'], best['seasonal_order'], best['model']
    print(f"✓ {nombre_serie}: Mejor SARIMA{order}x{sorder} AIC={best['aic']:.2f}")
    pred = res.get_prediction(start=y_test.index[0], end=y_test.index[-1], dynamic=False)
    y_pred = pred.predicted_mean
    ci = pred.conf_int()
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = _mape(y_test.values, y_pred.values)
    print(
        f"Resultados '{nombre_serie}' (TEST {y_test.index[0].date()} → {y_test.index[-1].date()}):"
    )
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    fc_res = res.get_forecast(steps=pasos_forecast)
    fc = fc_res.predicted_mean
    fc_ci = fc_res.conf_int()
    titulo = (
        f"{nombre_serie}\n"
        f"SARIMA{order}x{sorder} (AIC={best['aic']:.1f}) | Test MAPE={mape:.2f}%"
    )
    out_png = GRAFICAS_MODELADO_DIR / f"sarima_{nombre_serie.replace(' ', '_').lower()}.png"
    _plot_fit_vs_real(y, res.fittedvalues, fc, fc_ci, titulo, out_png)
    print(f"✓ Gráfica guardada: {out_png}")
    metrics_path = MODELOS_DIR / f"sarima_{nombre_serie.replace(' ', '_').lower()}_metrics.csv"
    pd.Series(
        {
            'serie': nombre_serie,
            'order': str(order),
            'seasonal_order': str(sorder),
            'aic': best['aic'],
            'mae': mae,
            'rmse': rmse,
            'mape_%': mape,
            'train_ini': y_train.index[0].strftime('%Y-%m'),
            'train_fin': y_train.index[-1].strftime('%Y-%m'),
            'test_ini': y_test.index[0].strftime('%Y-%m'),
            'test_fin': y_test.index[-1].strftime('%Y-%m'),
            'forecast_steps': pasos_forecast,
        }
    ).to_csv(metrics_path)
    return {
        'nombre_serie': nombre_serie,
        'order': order,
        'seasonal_order': sorder,
        'aic': best['aic'],
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'fitted': res.fittedvalues,
        'forecast': fc,
        'forecast_ci': fc_ci,
        'y_train': y_train,
        'y_test': y_test,
        'y_all': y,
    }

ENTITY_FILTERS = {
    'Skandia': "Skandia Afp - Accai S.A.",
    'Protección': '"Proteccion"',
    'Porvenir': '"Porvenir"',
    'Colfondos': '"Colfondos S.A." Y "Colfondos"',
}

FONDOS_POR_ENTIDAD = [
    ('Fondo de Cesantias Largo Plazo', 'cesantias_largo_plazo'),
    ('Fondo de Cesantias Corto Plazo', 'cesantias_corto_plazo'),
    ('Fondo de Pensiones Moderado', 'pensiones_moderado'),
    ('Fondo de Pensiones Conservador', 'pensiones_conservador'),
    ('Fondo de Pensiones Mayor Riesgo', 'pensiones_mayor_riesgo'),
    ('Fondo de Pensiones Retiro Programado', 'pensiones_retiro_programado'),
    ('Fondo de Pensiones Alternativo', 'pensiones_alternativo'),
]


@dataclass
class PensionesPipeline:
    base_url: str = BASE
    resource_id: str = RESOURCE
    df: pd.DataFrame = field(init=False, default=pd.DataFrame())
    entity_subsets: Dict[str, Dict[str, pd.DataFrame]] = field(init=False, default_factory=dict)
    resultados_modelado: Dict[str, dict] = field(init=False, default_factory=dict)
    resultados_arima: Dict[str, dict] = field(init=False, default_factory=dict)
    resultados_skandia: Dict[str, dict] = field(init=False, default_factory=dict)
    duplicados: int = field(init=False, default=0)
    duplicados_conceptuales: int = field(init=False, default=0)
    outliers: pd.DataFrame = field(init=False, default=pd.DataFrame())

    def run(self):
        ensure_directories()
        self.df = self._descargar_datos()
        self._limpieza_inicial()
        self._analisis_basico()
        self._exportar_referencias()
        self._verificar_relaciones()
        self._normalizar_textos()
        self._manejar_duplicados()
        self._analizar_outliers()
        self._optimizar_tipos()
        self._crear_variables_temporales()
        self._generar_subsets()
        self._validar_y_exportar()
        self._comparar_fondos_principales()
        self._graficas_entidad_y_correlaciones()
        self._crear_lags_iniciales()
        self._eda_general()
        self._exportar_dataset_modelado()
        self._crear_lags_optimizado()
        self._modelado_arima_basico()
        self._visualizar_modelado()
        self._regresiones_lineales()
        self._comparar_con_ipc()
        self._comparar_ipc_todos_fondos()
        self._cesantias_corto_plazo_base100()
        self._modelado_sarima()
        self._diferenciacion_colfondos()
        self._exportar_diccionario_columnas()
        self._analizar_nulos_tiempo()
        self._tablas_dinamicas()
        self._analisis_rendimientos()
        self._quartiles_rendimiento()
        self._cobertura_temporal()
        self._drawdown()
        print("\n=== PIPELINE COMPLETO EJECUTADO ===")

    # ------------------------------------------------------------------
    # Descarga y limpieza
    # ------------------------------------------------------------------
    def _descargar_datos(self) -> pd.DataFrame:
        print("Descargando datos desde API de datos.gov.co...")
        lista_paginas = []
        limit = 50000
        offset = 0
        while True:
            params = {"$limit": limit, "$offset": offset}
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            respuesta_json = r.json()
            if not respuesta_json:
                break
            lista_paginas.append(pd.DataFrame(respuesta_json))
            offset += limit
            print(f"Descargadas: {offset} filas…")
            time.sleep(0.3)
        if lista_paginas:
            df = pd.concat(lista_paginas, ignore_index=True)
        else:
            df = pd.DataFrame()
        try:
            total_filas = int(requests.get(f"{URL}?$select=count(*)").json()[0]["count"])
        except Exception:
            total_filas = None
        print("Total reportado:", total_filas)
        return df

    def _limpieza_inicial(self):
        df = self.df
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["valor_unidad"] = (
            df["valor_unidad"]
            .astype(str)
            .str.replace(r"[^\d\-,\.]", "", regex=True)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
        print(df.dtypes)
        df['valor_unidad'] = df['valor_unidad'].ffill()
        df['valor_unidad'] = df['valor_unidad'].interpolate()
        self.df = df

    def _analisis_basico(self):
        df = self.df
        nulls = df.isna().mean().sort_values(ascending=False).mul(100).round(2)
        print(nulls)
        cardinalidad = df.nunique(dropna=True).sort_values(ascending=False)
        print(cardinalidad)
        print("Valores únicos en nombre_entidad:", df["nombre_entidad"].dropna().unique()[:10])
        print("Valores únicos en nombre_fondo:", df["nombre_fondo"].dropna().unique()[:10])
        print("Conteo nombre_entidad:")
        print(df["nombre_entidad"].value_counts(dropna=False).head(10))
        print("Conteo nombre_fondo:")
        print(df["nombre_fondo"].value_counts(dropna=False).head(20))

    def _exportar_referencias(self):
        df = self.df
        df_clean = df.drop(columns=["codigo_entidad", "codigo_patrimonio"])
        df_clean.to_csv(RAW_DIR / "pensionesLimpio.csv", index=False)
        dict_entidad = (
            df[["nombre_entidad", "codigo_entidad"]]
            .drop_duplicates()
            .set_index("nombre_entidad")["codigo_entidad"]
            .to_dict()
        )
        dict_fondo = (
            df[["nombre_fondo", "codigo_patrimonio"]]
            .drop_duplicates()
            .set_index("nombre_fondo")["codigo_patrimonio"]
            .to_dict()
        )
        df[["nombre_entidad", "codigo_entidad"]].drop_duplicates().to_csv(
            RAW_DIR / "entidad_codigo.csv", index=False
        )
        df[["nombre_fondo", "codigo_patrimonio"]].drop_duplicates().to_csv(
            RAW_DIR / "fondos_codigo.csv", index=False
        )
        print("Diccionarios generados:", len(dict_entidad), len(dict_fondo))

    def _verificar_relaciones(self):
        df = self.df
        print("Relación código_entidad → nombre_entidad:")
        print(df.groupby("codigo_entidad")["nombre_entidad"].nunique().sort_values(ascending=False).head())
        print("Relación código_patrimonio → nombre_fondo:")
        print(df.groupby("codigo_patrimonio")["nombre_fondo"].nunique().sort_values(ascending=False).head())
        print("Relación nombre_entidad → código_entidad:")
        print(df.groupby("nombre_entidad")["codigo_entidad"].nunique().sort_values(ascending=False).head())
        print("Relación nombre_fondo → código_patrimonio:")
        print(df.groupby("nombre_fondo")["codigo_patrimonio"].nunique().sort_values(ascending=False).head())

    def _normalizar_textos(self):
        for c in ["nombre_entidad", "nombre_fondo"]:
            self.df[c] = (
                self.df[c]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
        print("Cardinalidad después de limpieza:")
        print(self.df[["nombre_entidad", "nombre_fondo"]].nunique())
        print("Valores únicos en nombre_entidad:", self.df["nombre_entidad"].unique())
        print("Valores únicos en nombre_fondo:", self.df["nombre_fondo"].unique()[:10])
        print("Conteo final nombre_entidad:")
        print(self.df["nombre_entidad"].value_counts())
        print("Conteo final nombre_fondo:")
        print(self.df["nombre_fondo"].value_counts().head(20))

    def _manejar_duplicados(self):
        df = self.df
        print("\n=== ANÁLISIS DE DUPLICADOS ===")
        self.duplicados = df.duplicated().sum()
        print(f"Filas duplicadas exactas: {self.duplicados}")
        if self.duplicados > 0:
            print("Eliminando duplicados exactos...")
            df = df.drop_duplicates()
            print(f"Dataset después de eliminar duplicados: {len(df)} filas")
        else:
            print("✓ No hay duplicados exactos")
        self.duplicados_conceptuales = df.duplicated(
            subset=['nombre_entidad', 'nombre_fondo', 'fecha']
        ).sum()
        print(f"Duplicados conceptuales (misma entidad-fondo-fecha): {self.duplicados_conceptuales}")
        if self.duplicados_conceptuales > 0:
            print("Manteniendo el primer registro de cada duplicado conceptual...")
            df = df.drop_duplicates(subset=['nombre_entidad', 'nombre_fondo', 'fecha'], keep='first')
            print(f"Dataset después de limpieza: {len(df)} filas")
        self.df = df

    def _analizar_outliers(self):
        df = self.df
        print("\n=== ANÁLISIS DE OUTLIERS EN valor_unidad ===")
        Q1 = df['valor_unidad'].quantile(0.25)
        Q3 = df['valor_unidad'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = df[(df['valor_unidad'] < limite_inferior) | (df['valor_unidad'] > limite_superior)]
        print(f"Límite inferior (outliers): {limite_inferior:.2f}")
        print(f"Límite superior (outliers): {limite_superior:.2f}")
        print(f"Total de outliers detectados: {len(outliers)}")
        if len(outliers) > 0:
            print("\nMuestra de outliers:")
            print(outliers[['nombre_entidad', 'nombre_fondo', 'fecha', 'valor_unidad']].head())
            df['es_outlier'] = (
                (df['valor_unidad'] < limite_inferior) | (df['valor_unidad'] > limite_superior)
            )
            print("✓ Columna 'es_outlier' creada para análisis posterior")
        else:
            df['es_outlier'] = False
            print("✓ No se detectaron outliers significativos")
        self.outliers = outliers
        self.df = df

    def _optimizar_tipos(self):
        df = self.df
        df['nombre_entidad'] = df['nombre_entidad'].astype('category')
        df['nombre_fondo'] = df['nombre_fondo'].astype('category')
        df['es_outlier'] = df['es_outlier'].astype('bool')
        self.df = df

    def _crear_variables_temporales(self):
        df = self.df
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter

        def clasificar_fondo(nombre_fondo):
            nombre = nombre_fondo.lower()
            if 'cesantia' in nombre:
                return 'Cesantías'
            if 'pension' in nombre:
                return 'Pensiones'
            if 'alternativo' in nombre:
                return 'Alternativo'
            return 'Otros'

        df['tipo_fondo'] = df['nombre_fondo'].apply(clasificar_fondo).astype('category')
        print("Variables derivadas creadas:")
        print(
            f"  - tipo_fondo: {df['tipo_fondo'].value_counts().to_dict()}"
        )
        self.df = df

    # ------------------------------------------------------------------
    # Subsets por entidad y exportaciones
    # ------------------------------------------------------------------
    def _generar_subsets(self):
        df = self.df
        subsets: Dict[str, Dict[str, pd.DataFrame]] = {}
        for entidad_display, filtro in ENTITY_FILTERS.items():
            slug = slugify(entidad_display)
            entidad_df = guardar_subset(
                df,
                "nombre_entidad",
                filtro,
                str(RAW_DIR / f"{slug}.csv"),
            )
            subset_detail = {'__entidad__': entidad_df}
            for fondo_nombre, fondo_slug in FONDOS_POR_ENTIDAD:
                subset_detail[fondo_nombre] = guardar_subset(
                    entidad_df,
                    "nombre_fondo",
                    fondo_nombre,
                    str(RAW_DIR / f"{slug}_fondo_{fondo_slug}.csv"),
                )
            subsets[entidad_display] = subset_detail
        self.entity_subsets = subsets

    def _validar_y_exportar(self):
        df = self.df
        print("\n=== VALIDACIÓN FINAL ===")
        print(f"Dimensiones finales del dataset: {df.shape}")
        print("Tipos de datos finales:")
        print(df.dtypes)
        memoria = df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"\nResumen de memoria utilizada: {memoria:.2f} MB")
        df.to_csv(PROCESSED_DIR / "pensiones_limpio_final.csv", index=False, encoding='utf-8')
        resumen_limpieza = {
            'filas_finales': len(df),
            'columnas_finales': len(df.columns),
            'duplicados_eliminados': self.duplicados,
            'outliers_detectados': len(self.outliers),
            'memoria_mb': memoria,
            'fecha_limpieza': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        pd.Series(resumen_limpieza).to_csv(PROCESSED_DIR / "resumen_limpieza.csv")
        print("✓ Dataset limpio exportado y resumen generado")

    # ------------------------------------------------------------------
    # Visualizaciones comparativas
    # ------------------------------------------------------------------
    def _comparar_fondos_principales(self):
        if not self.entity_subsets:
            return
        comparaciones = {
            "comparacion_pensiones_moderado": 'Fondo de Pensiones Moderado',
            "comparacion_pensiones_conservador": 'Fondo de Pensiones Conservador',
            "comparacion_cesantias_largo": 'Fondo de Cesantias Largo Plazo',
            "comparacion_cesantias_corto": 'Fondo de Cesantias Corto Plazo',
            "comparacion_pensiones_mayor_riesgo": 'Fondo de Pensiones Mayor Riesgo',
            "comparacion_pensiones_retiro_programado": 'Fondo de Pensiones Retiro Programado',
            "comparacion_pensiones_alternativo": 'Fondo de Pensiones Alternativo',
        }
        for archivo, fondo_nombre in comparaciones.items():
            fondos = {
                entidad: datos.get(fondo_nombre, pd.DataFrame())
                for entidad, datos in self.entity_subsets.items()
            }
            graficar_comparacion_entidades_por_fondo(
                fondos,
                fondo_nombre,
                archivo,
            )

    def _graficas_entidad_y_correlaciones(self):
        for entidad, fondos in self.entity_subsets.items():
            evolucion_todos_fondos_entidad(entidad, fondos)
            if entidad in ['Skandia', 'Protección', 'Porvenir', 'Colfondos']:
                matriz_correlacion_fondos(entidad, fondos)

    # ------------------------------------------------------------------
    # Lags y EDA
    # ------------------------------------------------------------------
    def _crear_lags_iniciales(self):
        df = self.df
        for lag in range(1, len(df) // 30, 30):
            df[f'lag_{lag}'] = df['valor_unidad'].shift(lag)
        self.df = df

    def _eda_general(self):
        df = self.df
        print("\n" + "=" * 60)
        print("ANÁLISIS EXPLORATORIO COMPLETO (EDA)")
        print("=" * 60)
        GRAFICAS_COMPARATIVAS_DIR.mkdir(parents=True, exist_ok=True)
        print("\n1. ESTADÍSTICAS DESCRIPTIVAS:")
        print(df[['valor_unidad']].describe())
        print("\nEstadísticas por tipo de fondo:")
        print(df.groupby('tipo_fondo', observed=True)['valor_unidad'].describe())
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        df['valor_unidad'].hist(bins=50, alpha=0.7)
        plt.title('Distribución de valor_unidad')
        plt.xlabel('Valor Unidad')
        plt.ylabel('Frecuencia')
        plt.subplot(1, 2, 2)
        df['valor_unidad'].plot(kind='density')
        plt.title('Densidad de valor_unidad')
        plt.xlabel('Valor Unidad')
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'distribucion_valor_unidad.png', dpi=300, bbox_inches='tight')
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'densidad_valor_unidad.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: distribucion_valor_unidad.png y densidad_valor_unidad.png")
        print("\n3. ANÁLISIS TEMPORAL:")
        evolucion_anual = df.groupby('año')['valor_unidad'].agg(['mean', 'std', 'min', 'max'])
        print("Evolución anual:")
        print(evolucion_anual)
        plt.figure(figsize=(12, 6))
        evolucion_anual['mean'].plot(kind='line', marker='o')
        plt.fill_between(
            evolucion_anual.index,
            evolucion_anual['mean'] - evolucion_anual['std'],
            evolucion_anual['mean'] + evolucion_anual['std'],
            alpha=0.2,
        )
        plt.title('Evolución Anual del Valor Unidad (media ± desviación)')
        plt.xlabel('Año')
        plt.ylabel('Valor Unidad')
        plt.grid(True, alpha=0.3)
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'evolucion_anual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: evolucion_anual.png")
        fondo_ejemplo = self.entity_subsets['Skandia']['Fondo de Pensiones Moderado'].set_index('fecha')['valor_unidad']
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plot_acf(fondo_ejemplo, lags=30, ax=plt.gca())
        plt.title('Autocorrelación (Fondo Moderado Skandia)')
        plt.subplot(1, 2, 2)
        plot_pacf(fondo_ejemplo, lags=30, ax=plt.gca())
        plt.title('Autocorrelación Parcial (Fondo Moderado Skandia)')
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'autocorrelacion.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: autocorrelacion.png")
        fondo_mensual = fondo_ejemplo.resample('ME').mean()
        try:
            descomposicion = seasonal_decompose(fondo_mensual, model='additive', period=12)
            fig = descomposicion.plot()
            fig.set_size_inches(12, 8)
            fig.suptitle('Descomposición Estacional - Fondo Moderado Skandia (Mensual)', fontsize=14)
            plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'descomposicion_estacional.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Descomposición estacional completada")
        except Exception as exc:
            print(f"✗ Error en descomposición estacional: {exc}")
        estacionalidad_mensual = df.groupby('mes')['valor_unidad'].mean()
        plt.figure(figsize=(10, 6))
        estacionalidad_mensual.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('Comportamiento Estacional Promedio por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Valor Unidad Promedio')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'estacionalidad_mensual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: estacionalidad_mensual.png")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='tipo_fondo', y='valor_unidad')
        plt.title('Distribución y Outliers por Tipo de Fondo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'boxplot_tipos_fondo.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: boxplot_tipos_fondo.png")
        fondo_ejemplo_vol = fondo_ejemplo.rolling(window=30).std()
        plt.figure(figsize=(12, 6))
        plt.plot(fondo_ejemplo_vol.index, fondo_ejemplo_vol.values, color='red', alpha=0.7)
        plt.title('Volatilidad Rolling (30 días) - Fondo Moderado Skandia')
        plt.xlabel('Fecha')
        plt.ylabel('Volatilidad (Desviación Estándar)')
        plt.grid(True, alpha=0.3)
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'volatilidad_rolling.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: volatilidad_rolling.png")
        pivot_corr = df.pivot_table(index='fecha', columns='tipo_fondo', values='valor_unidad', observed=False).corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlación entre Diferentes Tipos de Fondos')
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'correlacion_tipos_fondo.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: correlacion_tipos_fondo.png")
        tendencia_30d = fondo_ejemplo.rolling(window=30).mean()
        tendencia_90d = fondo_ejemplo.rolling(window=90).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(fondo_ejemplo.index, fondo_ejemplo.values, label='Valor Diario', alpha=0.3)
        plt.plot(tendencia_30d.index, tendencia_30d.values, label='Tendencia 30 días', linewidth=2)
        plt.plot(tendencia_90d.index, tendencia_90d.values, label='Tendencia 90 días', linewidth=2)
        plt.title('Análisis de Tendencia - Fondo Moderado Skandia')
        plt.xlabel('Fecha')
        plt.ylabel('Valor Unidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'analisis_tendencia.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: analisis_tendencia.png")
        plt.figure(figsize=(12, 6))
        plt.scatter(df.index, df['valor_unidad'], c=df['es_outlier'], cmap='coolwarm', alpha=0.6)
        plt.title('Identificación Visual de Outliers')
        plt.xlabel('Índice')
        plt.ylabel('Valor Unidad')
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'outliers_detallado.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: outliers_detallado.png")
        plt.figure(figsize=(12, 6))
        for tipo in df['tipo_fondo'].unique():
            subset = df[df['tipo_fondo'] == tipo]
            plt.hist(subset['valor_unidad'], bins=50, alpha=0.5, label=tipo, density=True)
        plt.title('Distribución de Densidad por Tipo de Fondo')
        plt.xlabel('Valor Unidad')
        plt.ylabel('Densidad')
        plt.legend()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'densidad_tipos_fondo.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: densidad_tipos_fondo.png")
        years = sorted(df['año'].unique())
        n_cols = 3
        n_rows = (len(years) + n_cols - 1) // n_cols
        EDA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(15, 5 * n_rows))
        for i, year in enumerate(years, 1):
            plt.subplot(n_rows, n_cols, i)
            data_year = df[df['año'] == year]
            for fondo in data_year['tipo_fondo'].unique():
                data_fondo = data_year[data_year['tipo_fondo'] == fondo]
                monthly_avg = data_fondo.groupby('mes')['valor_unidad'].mean()
                plt.plot(monthly_avg.index, monthly_avg.values, label=fondo, marker='o')
            plt.title(f'Evolución mensual {year}')
            plt.xlabel('Mes')
            plt.ylabel('Valor Unitario Promedio')
            plt.xticks(range(1, 13))
            plt.grid(True, alpha=0.3)
            if i == 1:
                plt.legend()
        plt.tight_layout()
        plt.savefig(EDA_IMAGES_DIR / 'evolucion_mensual_por_año.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfica guardada: evolucion_mensual_por_año.png")
        print("\n14. RESUMEN PARA MODELADO:")
        print(f"• Rango temporal: {df['fecha'].min()} to {df['fecha'].max()}")
        print(f"• Total de observaciones: {len(df):,}")
        print(f"• Número de entidades únicas: {df['nombre_entidad'].nunique()}")
        print(f"• Número de fondos únicos: {df['nombre_fondo'].nunique()}")
        print(f"• Rango de valores: {df['valor_unidad'].min():.2f} - {df['valor_unidad'].max():.2f}")
        print(f"• Coeficiente de variación: {(df['valor_unidad'].std() / df['valor_unidad'].mean() * 100):.2f}%")

    def _exportar_dataset_modelado(self):
        df_modelado = self.df.copy()
        df_modelado = df_modelado.dropna(subset=['valor_unidad', 'fecha', 'nombre_entidad', 'nombre_fondo'])
        df_modelado.to_csv(PROCESSED_DIR / "pensiones_listo_modelado.csv", index=False, encoding='utf-8')
        print("✓ Dataset listo para modelado exportado a data/processed/pensiones_listo_modelado.csv")

    def _crear_lags_optimizado(self):
        df = self.df
        try:
            lags = [1, 7, 30, 90, 180, 365]
            lag_columns = {}
            for lag in lags:
                lag_columns[f'lag_{lag}'] = df['valor_unidad'].shift(lag)
            lag_df = pd.DataFrame(lag_columns)
            self.df = pd.concat([df, lag_df], axis=1)
        except Exception as exc:
            print(f"✗ Error creando lags: {exc}")

    # ------------------------------------------------------------------
    # Modelado ARIMA básico y visualización
    # ------------------------------------------------------------------
    def _modelado_arima_basico(self):
        series_a_modelar = {
            "Fondo Moderado Skandia": self.entity_subsets['Skandia']['Fondo de Pensiones Moderado'],
            "Fondo Conservador Porvenir": self.entity_subsets['Porvenir']['Fondo de Pensiones Conservador'],
            "Cesantías Largo Plazo Colfondos": self.entity_subsets['Colfondos']['Fondo de Cesantias Largo Plazo'],
        }
        resultados = {}
        for nombre_serie, df_serie in series_a_modelar.items():
            if len(df_serie) > 100:
                try:
                    resultados[nombre_serie] = pipeline_modelado_completo(df_serie, nombre_serie)
                except Exception as exc:
                    print(f"✗ Error en modelado de {nombre_serie}: {exc}")
            else:
                print(f"⚠️  Serie {nombre_serie} muy corta para modelado ({len(df_serie)} registros)")
        self.resultados_modelado = resultados
        if resultados:
            comparacion_modelos = []
            for nombre, res in resultados.items():
                if res.get('mejor_modelo'):
                    comparacion_modelos.append(
                        {
                            'Serie': nombre,
                            'Mejor Modelo': f"ARIMA{res['mejor_modelo']['orden']}",
                            'AIC': res['mejor_modelo']['metricas']['aic'],
                            'Estacionaria': res['estacionariedad']['es_estacionaria'],
                            'MAPE (%)': res.get('evaluacion', {}).get('MAPE', 'N/A'),
                        }
                    )
            if comparacion_modelos:
                df_comparacion = pd.DataFrame(comparacion_modelos)
                print("\n📊 COMPARACIÓN DE MODELOS:")
                print(df_comparacion.to_string(index=False))
                MODELOS_DIR.mkdir(parents=True, exist_ok=True)
                df_comparacion.to_csv(MODELOS_DIR / "comparacion_modelos.csv", index=False)
                print("✓ Comparación de modelos guardada en: data/modelos/comparacion_modelos.csv")
                mejor_modelo = df_comparacion.loc[df_comparacion['AIC'].idxmin()]
                print(f"• Mejor modelo general: {mejor_modelo['Serie']} ({mejor_modelo['Mejor Modelo']})")
                print(f"• AIC más bajo: {mejor_modelo['AIC']:.2f}")
                print("\n💡 RECOMENDACIONES PARA PRÓXIMOS PASOS:")
                print("1. Para series estacionarias: Considerar modelos ARMA puros")
                print("2. Para series no estacionarias: Explorar diferenciación estacional (SARIMA)")
                print("3. Series con alto MAPE: Investigar outliers y eventos atípicos")
                print("4. Considerar modelos de machine learning (Random Forest, XGBoost) para comparación")

    def _visualizar_modelado(self):
        if self.resultados_modelado:
            visualizar_resultados_modelado(self.resultados_modelado)

    # ------------------------------------------------------------------
    # Regresiones lineales y comparaciones
    # ------------------------------------------------------------------
    def _regresiones_lineales(self):
        serie_cp = (
            self.entity_subsets['Skandia']['Fondo de Cesantias Corto Plazo'][['fecha', 'valor_unidad']]
            .dropna()
            .sort_values('fecha')
            .copy()
        )
        if serie_cp.empty:
            print("Sin datos para Skandia Cesantías Corto Plazo.")
        else:
            t0 = serie_cp['fecha'].min()
            X = (serie_cp['fecha'] - t0).dt.days.values.reshape(-1, 1)
            y = serie_cp['valor_unidad'].values
            lin = LinearRegression().fit(X, y)
            y_fit = lin.predict(X)
            inicio_linea = pd.Timestamp('2016-01-01')
            fin_datos = serie_cp['fecha'].max()
            fin_linea = max(fin_datos, pd.Timestamp('2026-12-31'))
            fechas_linea = pd.date_range(start=inicio_linea, end=fin_linea, freq='D')
            X_linea = ((fechas_linea - t0).days.values).reshape(-1, 1)
            y_linea = lin.predict(X_linea)
            plt.figure(figsize=(14, 6))
            plt.plot(serie_cp['fecha'], serie_cp['valor_unidad'], label='Real (Skandia Cesantías Corto Plazo)', alpha=0.85)
            plt.plot(fechas_linea, y_linea, '--', linewidth=2.2, label='Recta de regresión (2016–2026)')
            plt.title('Serie real vs. Recta de regresión lineal\nSkandia – Cesantías Corto Plazo')
            plt.xlabel('Fecha')
            plt.ylabel('Valor Unidad')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(GRAFICAS_COMPARATIVAS_DIR / 'reg_lineal_skandia_cesantias_corto_plazo.png', dpi=300, bbox_inches='tight')
            plt.close()
            mae = mean_absolute_error(y, y_fit)
            rmse = np.sqrt(mean_squared_error(y, y_fit))
            mape = np.mean(np.abs((y - y_fit) / y)) * 100
            print(f"Métricas (sobre la muestra): MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
        fondos_skandia = {
            nombre: datos
            for nombre, datos in self.entity_subsets['Skandia'].items()
            if nombre in dict(FONDOS_POR_ENTIDAD)
        }
        resultados_skandia = {}
        for nombre_fondo, df_fondo in fondos_skandia.items():
            if len(df_fondo) < 30:
                print(f"⚠️ Fondo '{nombre_fondo}' tiene muy pocos datos. Se omite.")
                continue
            df_temp = df_fondo[['fecha', 'valor_unidad']].dropna().sort_values('fecha').copy()
            t0 = df_temp['fecha'].min()
            X = (df_temp['fecha'] - t0).dt.days.values.reshape(-1, 1)
            y = df_temp['valor_unidad'].values
            modelo = LinearRegression().fit(X, y)
            y_fit = modelo.predict(X)
            ultima_fecha = df_temp['fecha'].max()
            fecha_futura = ultima_fecha + pd.Timedelta(days=365)
            dias_futuros = np.array([(fecha_futura - t0).days]).reshape(-1, 1)
            pred_futuro = modelo.predict(dias_futuros)[0]
            fechas_linea = pd.date_range(start=t0, end=fecha_futura, freq='D')
            X_linea = ((fechas_linea - t0).days.values).reshape(-1, 1)
            y_linea = modelo.predict(X_linea)
            mae = mean_absolute_error(y, y_fit)
            rmse = np.sqrt(mean_squared_error(y, y_fit))
            mape = np.mean(np.abs((y - y_fit) / y)) * 100
            resultados_skandia[nombre_fondo] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "Valor_predicho_1año": pred_futuro,
                "Última_fecha": ultima_fecha.strftime('%Y-%m-%d'),
                "Fecha_predicha": fecha_futura.strftime('%Y-%m-%d'),
            }
            plt.figure(figsize=(12, 6))
            plt.plot(df_temp['fecha'], df_temp['valor_unidad'], label='Real', color='blue', alpha=0.7)
            plt.plot(fechas_linea, y_linea, '--', label='Recta de regresión + pronóstico', color='orange')
            plt.axvline(ultima_fecha, color='gray', linestyle='--', alpha=0.5)
            plt.scatter(fecha_futura, pred_futuro, color='red', label=f'Predicción 1 año: {pred_futuro:.2f}')
            plt.title(f"Regresión lineal: {nombre_fondo} (Skandia)\nPredicción a 1 año")
            plt.xlabel("Fecha")
            plt.ylabel("Valor Unidad")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                GRAFICAS_MODELADO_DIR / f"reg_lineal_skandia_{slugify(nombre_fondo)}.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()
        self.resultados_skandia = resultados_skandia
        if resultados_skandia:
            df_resultados_skandia = pd.DataFrame(resultados_skandia).T
            df_resultados_skandia.to_csv(MODELOS_DIR / "predicciones_skandia_1año.csv", encoding='utf-8')
            print("✓ Resultados guardados en: data/modelos/predicciones_skandia_1año.csv")
            print("✓ Gráficas guardadas en: data/graficas_modelado/")

    # ------------------------------------------------------------------
    # Comparaciones IPC
    # ------------------------------------------------------------------
    def _comparar_con_ipc(self):
        import_path = DATA_DIR / "API_FP.CPI.TOTL_DS2_es_csv_v2_59981.csv"
        ipc_raw = pd.read_csv(import_path, skiprows=4)
        col_country = next(
            c for c in ipc_raw.columns if c.lower().startswith(("country name", "nombre del país", "nombre del pais"))
        )
        ipc_col = ipc_raw.loc[ipc_raw[col_country].str.lower() == "colombia"].copy()
        def es_col_anio(c):
            return re.match(r"^\d{4}($|\s|\[)", str(c)) is not None
        year_cols = [c for c in ipc_col.columns if es_col_anio(c)]
        rename_map = {c: int(re.match(r"^(\d{4})", str(c)).group(1)) for c in year_cols}
        ipc_col = ipc_col[year_cols].rename(columns=rename_map)
        ipc_col = ipc_col.apply(pd.to_numeric, errors="coerce")
        anio_ini, anio_fin_deseado = 2016, 2025
        anios_disp = sorted(ipc_col.columns.dropna())
        anio_fin = min(anio_fin_deseado, anios_disp[-1])
        anios_sel = [a for a in anios_disp if anio_ini <= a <= anio_fin]
        ipc_series = ipc_col.loc[:, anios_sel].T.squeeze()
        ipc_series.index = pd.to_datetime(ipc_series.index.astype(str) + "-12-31")
        ipc_series.name = "IPC Colombia (Base 100)"
        self.ipc_series = ipc_series
        sk_mod = self.entity_subsets['Skandia']['Fondo de Pensiones Moderado'].set_index("fecha")["valor_unidad"].sort_index()
        def base100(s):
            s = s.dropna()
            return s / s.iloc[0] * 100
        sk_b100_diario = base100(sk_mod[sk_mod.index >= "2016-01-01"])
        sk_b100_anual = sk_b100_diario.resample("A-DEC").last()
        ipc_b100 = base100(ipc_series[ipc_series.index >= "2016-01-01"])
        cmp = pd.concat(
            [ipc_b100.rename("IPC (base=100)"), sk_b100_anual.rename("Skandia Moderado (base=100)")],
            axis=1,
        ).sort_index().interpolate(limit_direction="both")
        plt.figure(figsize=(14, 6))
        plt.plot(cmp.index, cmp["Skandia Moderado (base=100)"], label="Fondo Skandia Moderado (Base 100)", linewidth=2)
        plt.plot(cmp.index, cmp["IPC (base=100)"], "--", label="IPC Colombia (Base 100)", linewidth=2, marker="o")
        plt.title("Comparación: Evolución del IPC vs. Fondo de Pensiones Moderado (Skandia)")
        plt.xlabel("Fecha")
        plt.ylabel("Índice Normalizado (Base 100 en 2016)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "ipc_vs_skandia_moderado.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _comparar_ipc_todos_fondos(self):
        if not hasattr(self, 'ipc_series'):
            print("No se cargó serie IPC. Saltando comparaciones adicionales.")
            return
        ipc_b100 = (self.ipc_series[self.ipc_series.index >= "2016-01-01"] / self.ipc_series[self.ipc_series.index >= "2016-01-01"].iloc[0] * 100).rename("IPC (base=100)")
        for nombre, df_fondo in self.entity_subsets['Skandia'].items():
            if nombre == '__entidad__':
                continue
            s = df_fondo.set_index("fecha")["valor_unidad"].sort_index().dropna()
            s = s[s.index >= "2016-01-01"]
            if s.empty:
                continue
            s_anual = s.resample("A-DEC").last()
            s_b100a = (s_anual / s_anual.iloc[0] * 100).rename(f"{nombre} (base=100)")
            cmp = pd.concat([ipc_b100, s_b100a], axis=1).sort_index().interpolate(limit_direction="both")
            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            ax.plot(cmp.index, cmp[f"{nombre} (base=100)"], label=f"{nombre} (Base 100)", linewidth=2)
            ax.plot(cmp.index, cmp["IPC (base=100)"], "--", label="IPC Colombia (Base 100)", linewidth=2, marker="o")
            ax.set_ylabel("Índice normalizado (Base 100 en 2016)")
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(s_anual.index, s_anual.values, ":", linewidth=1.8, color="gray", label="Valor unidad (anual, eje der.)")
            ax2.set_ylabel("Valor unidad (anual)")
            plt.title(f"IPC vs. {nombre} (Skandia)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left")
            plt.tight_layout()
            outfile = GRAFICAS_COMPARATIVAS_DIR / f"ipc_vs_skandia_{slugify(nombre)}.png"
            plt.savefig(outfile, dpi=300, bbox_inches="tight")
            plt.close()

    def _cesantias_corto_plazo_base100(self):
        if not hasattr(self, 'ipc_series'):
            return
        fondos_corto_dict = {
            "Skandia": self.entity_subsets['Skandia']['Fondo de Cesantias Corto Plazo'],
            "Protección": self.entity_subsets['Protección']['Fondo de Cesantias Corto Plazo'],
            "Porvenir": self.entity_subsets['Porvenir']['Fondo de Cesantias Corto Plazo'],
            "Colfondos": self.entity_subsets['Colfondos']['Fondo de Cesantias Corto Plazo'],
            "IPC": (
                self.ipc_series[self.ipc_series.index.year <= 2024]
                .rename("valor_unidad")
                .reset_index()
                .rename(columns={"index": "fecha"})
            ),
        }
        def _valor_anual_base100(df_fondo, nombre):
            if df_fondo is None or len(df_fondo) == 0:
                print(f"⚠️  Sin datos para: {nombre}")
                return None
            s = (
                df_fondo
                .dropna(subset=['fecha', 'valor_unidad'])
                .set_index('fecha')['valor_unidad']
                .sort_index()
            )
            s_anual = s.resample('A-DEC').last()
            s_anual = s_anual[s_anual.index.year <= 2024]
            if s_anual.empty:
                print(f"⚠️  Serie anual vacía (<=2024) para: {nombre}")
                return None
            s_anual.index = s_anual.index.year
            s_anual.name = nombre
            return s_anual
        series_anuales = {}
        for nombre, df_f in fondos_corto_dict.items():
            serie = _valor_anual_base100(df_f, nombre)
            if serie is not None:
                series_anuales[nombre] = serie
        if not series_anuales:
            return
        anios_comunes = set.intersection(*[set(s.index) for s in series_anuales.values()])
        anios_comunes = sorted([a for a in anios_comunes if 2016 <= a <= 2024])
        if len(anios_comunes) < 2:
            return
        anio_base = anios_comunes[0]
        def _base100_mismo_base(s, base_year, anios):
            s = s.loc[anios].copy()
            base_val = s.loc[base_year]
            return (s / base_val) * 100
        df_norm = pd.DataFrame({
            nombre: _base100_mismo_base(s, anio_base, anios_comunes) for nombre, s in series_anuales.items()
        })
        df_norm.index.name = "Año"
        plt.figure(figsize=(12, 6))
        for nombre in df_norm.columns:
            if nombre != "IPC":
                plt.plot(df_norm.index, df_norm[nombre], marker='o', linewidth=1.8, alpha=0.9, label=nombre)
        if "IPC" in df_norm.columns:
            plt.plot(df_norm.index, df_norm["IPC"], linestyle='--', linewidth=3.0, marker='s', color='black', label="IPC (Base 100)", zorder=10)
        plt.title(f"Cesantías Corto Plazo — Valor anual (base 100 en {anio_base})")
        plt.xlabel("Año")
        plt.ylabel("Índice normalizado")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = GRAFICAS_COMPARATIVAS_DIR / "cesantias_corto_plazo_valor_anual_base100_2016_2024.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        out_csv = MODELOS_DIR / "cesantias_corto_plazo_valor_anual_base100_2016_2024.csv"
        df_norm.round(2).to_csv(out_csv, encoding="utf-8")

    # ------------------------------------------------------------------
    # SARIMA y diferenciación adicional
    # ------------------------------------------------------------------
    def _modelado_sarima(self):
        series_a_modelar_nuevas = {
            "Cesantías Corto Plazo - Skandia": self.entity_subsets['Skandia']['Fondo de Cesantias Corto Plazo'],
            "Cesantías Corto Plazo - Protección": self.entity_subsets['Protección']['Fondo de Cesantias Corto Plazo'],
            "Cesantías Corto Plazo - Porvenir": self.entity_subsets['Porvenir']['Fondo de Cesantias Corto Plazo'],
            "Cesantías Corto Plazo - Colfondos": self.entity_subsets['Colfondos']['Fondo de Cesantias Corto Plazo'],
        }
        resultados_arima = {}
        for nombre, dfx in series_a_modelar_nuevas.items():
            if dfx is None or len(dfx) == 0:
                print(f"⚠️ Sin datos para {nombre}")
                continue
            try:
                res = entrenar_y_graficar_sarima(
                    dfx,
                    nombre,
                    meses_test=24,
                    pasos_forecast=12,
                    pdq=[(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2)],
                    spdq=[(0, 0, 0, 12), (1, 0, 0, 12), (0, 1, 1, 12), (1, 1, 1, 12)],
                )
                if res:
                    resultados_arima[nombre] = res
            except Exception as exc:
                print(f"✗ Error modelando {nombre}: {exc}")
        self.resultados_arima = resultados_arima

    def _diferenciacion_colfondos(self):
        df_series = self.entity_subsets['Colfondos']['Fondo de Cesantias Largo Plazo']
        serie_colfondos_clp = (
            df_series[['fecha', 'valor_unidad']]
            .dropna(subset=['fecha', 'valor_unidad'])
            .sort_values('fecha')
            .set_index('fecha')['valor_unidad']
            .astype(float)
        )
        serie_diff1 = serie_colfondos_clp.diff(1).dropna()
        serie_diff2 = serie_diff1.diff(1).dropna()
        serie_diff3 = serie_diff2.diff(1).dropna()
        print("\n=== DIFERENCIACIÓN REGULAR (ARIMA) PARA COLFONDOS CESANTÍAS LARGO PLAZO ===")
        analizar_estacionariedad(serie_colfondos_clp, "Colfondos CLP - Nivel")
        analizar_estacionariedad(serie_diff1, "Colfondos CLP - Diferencia 1 (d=1)")
        analizar_estacionariedad(serie_diff2, "Colfondos CLP - Diferencia 2 (d=2)")
        analizar_estacionariedad(serie_diff3, "Colfondos CLP - Diferencia 3 (d=3)")
        def _plot_series(data, titulo, ylabel, filename):
            plt.figure(figsize=(12, 4))
            plt.plot(data, linewidth=1.3)
            plt.title(titulo)
            plt.xlabel("Tiempo")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(GRAFICAS_MODELADO_DIR / filename, dpi=300, bbox_inches="tight")
            plt.close()
        _plot_series(serie_colfondos_clp, "Colfondos – Fondo de Cesantías Largo Plazo\nSerie original (nivel)", "Valor de la unidad", "colfondos_clp_nivel.png")
        _plot_series(serie_diff1, "Colfondos – Cesantías Largo Plazo\nDiferenciación 1 vez (d = 1)", "ΔXₜ", "colfondos_clp_diff1.png")
        _plot_series(serie_diff2, "Colfondos – Cesantías Largo Plazo\nDiferenciación 2 veces (d = 2)", "Δ²Xₜ", "colfondos_clp_diff2.png")
        _plot_series(serie_diff3, "Colfondos – Cesantías Largo Plazo\nDiferenciación 3 veces (d = 3)", "Δ³Xₜ", "colfondos_clp_diff3.png")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes[0, 0].plot(serie_colfondos_clp, linewidth=1.2)
        axes[0, 0].set_title("Serie original\nColfondos – Cesantías Largo Plazo")
        axes[0, 1].plot(serie_diff1, linewidth=1.2)
        axes[0, 1].set_title("Diferenciación 1 vez (d = 1)")
        axes[1, 0].plot(serie_diff2, linewidth=1.2)
        axes[1, 0].set_title("Diferenciación 2 veces (d = 2)")
        axes[1, 1].plot(serie_diff3, linewidth=1.2)
        axes[1, 1].set_title("Diferenciación 3 veces (d = 3)")
        for ax in axes.flatten():
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(GRAFICAS_MODELADO_DIR / "colfondos_clp_differencing_2x2.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Reportes adicionales
    # ------------------------------------------------------------------
    def _exportar_diccionario_columnas(self):
        resumen_columnas = pd.DataFrame({
            "tipo": self.df.dtypes.astype(str),
            "n_null": self.df.isna().sum(),
            "%_null": self.df.isna().mean().mul(100).round(2),
            "n_unique": self.df.nunique(),
        })
        resumen_columnas.to_csv(PROCESSED_DIR / "resumen_columnas.csv")

    def _analizar_nulos_tiempo(self):
        nulls_por_mes = (
            self.df.set_index("fecha")
            .isna()
            .resample("M")
            .mean()
            .mul(100)
        )
        nulls_por_mes['valor_unidad'].plot(figsize=(10, 4))
        plt.title("% de nulos en valor_unidad a lo largo del tiempo")
        plt.ylabel("% nulos")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "nulos_valor_unidad_tiempo.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _tablas_dinamicas(self):
        tabla_entidad_anio = pd.pivot_table(
            self.df,
            values="valor_unidad",
            index="año",
            columns="nombre_entidad",
            aggfunc="mean",
        )
        tabla_entidad_anio.to_csv(PROCESSED_DIR / "tabla_entidad_anio_mean_valor_unidad.csv")
        plt.figure(figsize=(12, 6))
        sns.heatmap(tabla_entidad_anio, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Valor Unidad Promedio por Entidad y Año")
        plt.ylabel("Año")
        plt.xlabel("Entidad")
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "heatmap_valor_unidad_entidad_anio.png", dpi=300, bbox_inches="tight")
        plt.close()
        tabla_tipo_anio = pd.pivot_table(
            self.df,
            values="valor_unidad",
            index="año",
            columns="tipo_fondo",
            aggfunc=["mean", "std"],
        )
        tabla_tipo_anio.to_csv(PROCESSED_DIR / "tabla_tipo_anio_mean_std.csv")
        plt.figure(figsize=(12, 6))
        sns.heatmap(tabla_tipo_anio["mean"], annot=True, fmt=".2f", cmap="YlOrBr")
        plt.title("Valor Unidad Promedio por Tipo de Fondo y Año")
        plt.ylabel("Año")
        plt.xlabel("Tipo de Fondo")
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "heatmap_valor_unidad_tipo_anio.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _analisis_rendimientos(self):
        fondo = self.entity_subsets['Skandia']['Fondo de Pensiones Moderado'].set_index("fecha")["valor_unidad"].sort_index()
        returns = fondo.pct_change().dropna()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        returns.hist(bins=50)
        plt.title("Histograma rendimientos diarios\nSkandia Moderado")
        plt.xlabel("rendimiento diario")
        plt.ylabel("frecuencia")
        plt.subplot(1, 2, 2)
        returns.plot(kind="density")
        plt.title("Densidad rendimientos diarios\nSkandia Moderado")
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "rendimientos_skandia_moderado.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _quartiles_rendimiento(self):
        df_returns = (
            self.df.set_index("fecha")
            .groupby(["nombre_entidad", "nombre_fondo"])
            ["valor_unidad"]
            .resample("A-DEC")
            .last()
            .groupby(level=[0, 1])
            .pct_change()
            .reset_index(name="rend_anual")
        )
        df_2023 = df_returns[df_returns["fecha"].dt.year == 2023].dropna(subset=["rend_anual"])
        df_2023["quartil_rend"] = pd.qcut(df_2023["rend_anual"], 4, labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"])
        tabla_quartiles = pd.crosstab(df_2023["nombre_entidad"], df_2023["quartil_rend"], normalize="index").round(2)
        tabla_quartiles.to_csv(PROCESSED_DIR / "entidad_por_quartil_rend_2023.csv")
        tabla_quartiles.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
        plt.title("Distribución de Fondos por Quartil de Rendimiento Anual (2023)")
        plt.xlabel("Entidad")
        plt.ylabel("Proporción de Fondos")
        plt.legend(title="Quartil de Rendimiento")
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "entidad_por_quartil_rend_2023.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _cobertura_temporal(self):
        self.df["anio"] = self.df["fecha"].dt.year
        tabla_cobertura = (
            self.df.groupby(["nombre_entidad", "nombre_fondo", "anio"])
            .size()
            .unstack("anio")
            .fillna(0)
        )
        if "Skandia Afp - Accai S.A." in tabla_cobertura.index:
            tabla_skandia_cob = tabla_cobertura.loc["Skandia Afp - Accai S.A."]
            plt.figure(figsize=(12, 6))
            sns.heatmap((tabla_skandia_cob > 0).astype(int), cmap="Greys", cbar=False)
            plt.title("Cobertura temporal (existencia de datos)\nFondos Skandia vs años")
            plt.xlabel("Año")
            plt.ylabel("Fondo")
            plt.tight_layout()
            plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "cobertura_fondos_skandia.png", dpi=300, bbox_inches="tight")
            plt.close()

    def _drawdown(self):
        fondo = self.entity_subsets['Skandia']['Fondo de Pensiones Moderado'].set_index("fecha")["valor_unidad"].sort_index()
        cummax = fondo.cummax()
        drawdown = (fondo - cummax) / cummax
        plt.figure(figsize=(12, 4))
        drawdown.plot()
        plt.title("Drawdown Skandia Moderado")
        plt.ylabel("drawdown (proporción)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(GRAFICAS_COMPARATIVAS_DIR / "drawdown_skandia_moderado.png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    pipeline = PensionesPipeline()
    pipeline.run()
