import pandas as pd
import requests
import time
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from scipy.stats import entropy as shannon_entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose 

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import month_plot, quarter_plot


BASE = "https://www.datos.gov.co"
RESOURCE = "uawh-cjvi"  # ID del dataset
URL = f"{BASE}/resource/{RESOURCE}.json"

try:
    total_filas = int(requests.get(f"{URL}?$select=count(*)").json()[0]["count"])
except Exception:
    total_filas  = None
print("Total reportado:", total_filas )

Lista_paginas = []
limit = 50000         
offset = 0
while True: 
    params = {"$limit": limit, "$offset": offset}
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    respuestaJson = r.json()
    if not respuestaJson: # fin de datos
        break
    Lista_paginas.append(pd.DataFrame(respuestaJson))
    offset += limit
    print(f"Descargadas: {offset} filas…")
    time.sleep(0.3)     # pequeña pausa para no saturar

if Lista_paginas:  # Si la lista NO está vacía
    df = pd.concat(Lista_paginas, ignore_index=True)
else:              # Si la lista está vacía
    df = pd.DataFrame()

# fechas (ajusta el nombre si difiere)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# numérico (quita comas/puntos si vienen como texto con separadores)
df["valor_unidad"] = (
    df["valor_unidad"]
      .astype(str)
      .str.replace(r"[^\d\-,\.]", "", regex=True)  # limpia símbolos
      .str.replace(",", ".", regex=False)          # si usan coma decimal
      .astype(float)
)

print(df.dtypes)
df.isnull().sum()
df['valor_unidad'] = df['valor_unidad'].ffill()
df['valor_unidad'] = df['valor_unidad'].interpolate()
# % de nulos por columna
nulls = df.isna().mean().sort_values(ascending=False).mul(100).round(2)
print(nulls)

cardinalidad = df.nunique(dropna=True).sort_values(ascending=False)
print(cardinalidad)

# valores únicos (muestra)
print("Valores únicos en nombre_entidad:", df["nombre_entidad"].dropna().unique()[:10])
print("Valores únicos en nombre_fondo:", df["nombre_fondo"].dropna().unique()[:10])
print("Conteo nombre_entidad:")
print(df["nombre_entidad"].value_counts(dropna=False).head(10))
print("Conteo nombre_fondo:")
print(df["nombre_fondo"].value_counts(dropna=False).head(20))

df_clean = df.drop(columns=["codigo_entidad", "codigo_patrimonio"])
df_clean.to_csv("data/raw/pensionesLimpio.csv", index=False)

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

df[["nombre_entidad", "codigo_entidad"]].drop_duplicates() \
   .to_csv("data/raw/entidad_codigo.csv", index=False)

df[["nombre_fondo", "codigo_patrimonio"]].drop_duplicates() \
   .to_csv("data/raw/fondos_codigo.csv", index=False)

# cuántos names por código (debe ser 1 si es one-to-one)
print("Relación código_entidad → nombre_entidad:")
print(df.groupby("codigo_entidad")["nombre_entidad"].nunique().sort_values(ascending=False).head())
print("Relación código_patrimonio → nombre_fondo:")
print(df.groupby("codigo_patrimonio")["nombre_fondo"].nunique().sort_values(ascending=False).head())

# y al revés: cuántos códigos por nombre
print("Relación nombre_entidad → código_entidad:")
print(df.groupby("nombre_entidad")["codigo_entidad"].nunique().sort_values(ascending=False).head())
print("Relación nombre_fondo → código_patrimonio:")
print(df.groupby("nombre_fondo")["codigo_patrimonio"].nunique().sort_values(ascending=False).head())

# Normalización de textos
for c in ["nombre_entidad", "nombre_fondo"]:
    df[c] = (df[c]
             .astype(str)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True))  

print("Cardinalidad después de limpieza:")
print(df[["nombre_entidad","nombre_fondo"]].nunique())

print("Valores únicos en nombre_entidad:", df["nombre_entidad"].unique())
print("Valores únicos en nombre_fondo:", df["nombre_fondo"].unique()[:10])

print("Conteo final nombre_entidad:")
print(df["nombre_entidad"].value_counts())
print("Conteo final nombre_fondo:")
print(df["nombre_fondo"].value_counts().head(20))

# =============================================================================
# ELIMINACIÓN DE DUPLICADOS (Complemento a tu análisis existente)
# =============================================================================

print("\n=== ANÁLISIS DE DUPLICADOS ===")
duplicados = df.duplicated().sum()
print(f"Filas duplicadas exactas: {duplicados}")

if duplicados > 0:
    print("Eliminando duplicados exactos...")
    df = df.drop_duplicates()
    print(f"Dataset después de eliminar duplicados: {len(df)} filas")
else:
    print("✓ No hay duplicados exactos")

# Buscar duplicados conceptuales (misma entidad, mismo fondo, misma fecha)
duplicados_conceptuales = df.duplicated(
    subset=['nombre_entidad', 'nombre_fondo', 'fecha']
).sum()

print(f"Duplicados conceptuales (misma entidad-fondo-fecha): {duplicados_conceptuales}")

if duplicados_conceptuales > 0:
    print("Manteniendo el primer registro de cada duplicado conceptual...")
    df = df.drop_duplicates(
        subset=['nombre_entidad', 'nombre_fondo', 'fecha'], 
        keep='first'
    )
    print(f"Dataset después de limpieza: {len(df)} filas")

# =============================================================================
# DETECCIÓN Y MANEJO DE OUTLIERS 
# =============================================================================

print("\n=== ANÁLISIS DE OUTLIERS EN valor_unidad ===")

# Estadísticas robustas para detección de outliers
Q1 = df['valor_unidad'].quantile(0.25)
Q3 = df['valor_unidad'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[
    (df['valor_unidad'] < limite_inferior) | 
    (df['valor_unidad'] > limite_superior)
]

print(f"Límite inferior (outliers): {limite_inferior:.2f}")
print(f"Límite superior (outliers): {limite_superior:.2f}")
print(f"Total de outliers detectados: {len(outliers)}")
print(f"Porcentaje de outliers: {(len(outliers)/len(df)*100):.2f}%")

if len(outliers) > 0:
    print("\nMuestra de outliers:")
    print(outliers[['nombre_entidad', 'nombre_fondo', 'fecha', 'valor_unidad']].head())
    
    # Crear columna flag para outliers
    df['es_outlier'] = (
        (df['valor_unidad'] < limite_inferior) | 
        (df['valor_unidad'] > limite_superior)
    )
    
    print("✓ Columna 'es_outlier' creada para análisis posterior")
else:
    df['es_outlier'] = False
    print("✓ No se detectaron outliers significativos")

# =============================================================================
# OPTIMIZACIÓN DE TIPOS DE DATOS (Complemento a tu limpieza)
# =============================================================================

print("\nOptimizando tipos de datos...")
df['nombre_entidad'] = df['nombre_entidad'].astype('category')
df['nombre_fondo'] = df['nombre_fondo'].astype('category')
df['es_outlier'] = df['es_outlier'].astype('bool')

print("✓ Columnas convertidas a categoría para optimización")

# =============================================================================
# CREACIÓN DE VARIABLES DERIVADAS (Nuevo - Para análisis posterior)
# =============================================================================

print("\n=== CREACIÓN DE VARIABLES DERIVADAS ===")

# Extraer componentes de fecha para análisis temporal
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter

# Clasificar fondos por tipo (nueva variable categórica)
def clasificar_fondo(nombre_fondo):
    nombre = nombre_fondo.lower()
    if 'cesantia' in nombre:
        return 'Cesantías'
    elif 'pension' in nombre:
        return 'Pensiones'
    elif 'alternativo' in nombre:
        return 'Alternativo'
    else:
        return 'Otros'

df['tipo_fondo'] = df['nombre_fondo'].apply(clasificar_fondo).astype('category')

print("Variables derivadas creadas:")
print(f"  - año, mes, trimestre: para análisis temporal")
print(f"  - tipo_fondo: {df['tipo_fondo'].value_counts().to_dict()}")

# =============================================================================
# GUARDAR SUBSETS (Tu código original mantenido)
# =============================================================================

Path("data/raw").mkdir(parents=True, exist_ok=True)

def guardar_subset(df, filtro, valores, salida):
    if isinstance(valores, (list, tuple, set)):
        df_subset = df.loc[df[filtro].isin(valores)].copy()
    else:
        df_subset = df.loc[df[filtro].eq(valores)].copy()
    if filtro in df_subset.columns:
        df_subset = df_subset.drop(columns=[filtro])  
    
    print(df_subset.shape)
    df_subset.to_csv(salida, index=False)
    return df_subset
 
#-----------------------------------------------------------------------------
df_skandia = guardar_subset(df, "nombre_entidad",
               "Skandia Afp - Accai S.A.",
               "data/raw/skandia.csv")
df_skandia_cesantias_largo_plazo = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/skandia_fondo_cesantias_largo_plazo.csv")
df_skandia_cesantias_corto_plazo = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/skandia_fondo_cesantias_corto_plazo.csv")
df_skandia_pensiones_moderado = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/skandia_fondo_pensiones_moderado.csv")
df_skandia_pensiones_conservador = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/skandia_fondo_pensiones_conservador.csv")
df_skandia_pensiones_mayor_riesgo = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/skandia_fondo_pensiones_mayor_riesgo.csv")
df_skandia_pensiones_retiro_programado = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/skandia_fondo_pensiones_retiro_programado.csv")
df_skandia_pensiones_alternativo = guardar_subset(df_skandia, "nombre_fondo",
               "Fondo de Pensiones Alternativo",
               "data/raw/skandia_fondo_pensiones_alternativo.csv")
#-----------------------------------------------------------------------------
df_proteccion =  guardar_subset(df, "nombre_entidad",
               '"Proteccion"',
               "data/raw/proteccion.csv")
df_proteccion_cesantias_largo_plazo = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/proteccion_fondo_cesantias_largo_plazo.csv")
df_proteccion_cesantias_corto_plazo = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/proteccion_fondo_cesantias_corto_plazo.csv")
df_proteccion_pensiones_moderado = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/proteccion_fondo_pensiones_moderado.csv")
df_proteccion_pensiones_conservador = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/proteccion_fondo_pensiones_conservador.csv")
df_proteccion_pensiones_mayor_riesgo = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/proteccion_fondo_pensiones_mayor_riesgo.csv")
df_proteccion_pensiones_retiro_programado = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/proteccion_fondo_pensiones_retiro_programado.csv")
df_proteccion_pensiones_alternativo = guardar_subset(df_proteccion, "nombre_fondo",
               "Fondo de Pensiones Alternativo",
               "data/raw/proteccion_fondo_pensiones_alternativo.csv",)

#-----------------------------------------------------------------------------

df_porvenir = guardar_subset(df, "nombre_entidad",
               '"Porvenir"',
               "data/raw/porvenir.csv")
df_porvenir_cesantias_largo_plazo = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/porvenir_fondo_cesantias_largo_plazo.csv")
df_porvenir_cesantias_corto_plazo = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/porvenir_fondo_cesantias_corto_plazo.csv")
df_porvenir_pensiones_moderado = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/porvenir_fondo_pensiones_moderado.csv")
df_porvenir_pensiones_conservador = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/porvenir_fondo_pensiones_conservador.csv")
df_porvenir_pensiones_mayor_riesgo = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/porvenir_fondo_pensiones_mayor_riesgo.csv")
df_porvenir_pensiones_retiro_programado = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/porvenir_fondo_pensiones_retiro_programado.csv")
df_porvenir_pensiones_alternativo = guardar_subset(df_porvenir, "nombre_fondo",
               "Fondo de Pensiones Alternativo",
               "data/raw/porvenir_fondo_pensiones_alternativo.csv")

#-----------------------------------------------------------------------------

df_colfondos = guardar_subset(df, "nombre_entidad",
               '"Colfondos S.A." Y "Colfondos"',
               "data/raw/colfondos.csv")
df_colfondos_cesantias_largo_plazo = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/colfondos_fondo_cesantias_largo_plazo.csv")
df_colfondos_cesantias_corto_plazo = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/colfondos_fondo_cesantias_corto_plazo.csv")
df_colfondos_pensiones_moderado = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/colfondos_fondo_pensiones_moderado.csv")
df_colfondos_pensiones_conservador = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/colfondos_fondo_pensiones_conservador.csv")
df_colfondos_pensiones_mayor_riesgo = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/colfondos_fondo_pensiones_mayor_riesgo.csv")
df_colfondos_pensiones_retiro_programado = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/colfondos_fondo_pensiones_retiro_programado.csv")
df_colfondos_pensiones_alternativo = guardar_subset(df_colfondos, "nombre_fondo",
               "Fondo de Pensiones Alternativo",
               "data/raw/colfondos_fondo_pensiones_alternativo.csv")


# =============================================================================
# VALIDACIÓN FINAL Y EXPORTACIÓN
# =============================================================================

print("\n=== VALIDACIÓN FINAL ===")
print(f"Dimensiones finales del dataset: {df.shape}")
print(f"Tipos de datos finales:")
print(df.dtypes)
print(f"\nResumen de memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Exportar dataset limpio y listo para análisis
df.to_csv("data/processed/pensiones_limpio_final.csv", index=False, encoding='utf-8')

print("✓ Dataset limpio exportado a: data/processed/pensiones_limpio_final.csv")

# Exportar resumen de limpieza
resumen_limpieza = {
    'filas_finales': len(df),
    'columnas_finales': len(df.columns),
    'duplicados_eliminados': duplicados,
    'outliers_detectados': len(outliers),
    'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2,
    'fecha_limpieza': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

pd.Series(resumen_limpieza).to_csv("data/processed/resumen_limpieza.csv")

print("✓ Resumen de limpieza exportado a: data/processed/resumen_limpieza.csv")


def graficar_comparacion_entidades_por_fondo(fondos_a_comparar, titulo_base, nombre_archivo):
   
    plt.figure(figsize=(14, 8))
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (entidad, df_fondo) in enumerate(fondos_a_comparar.items()):
        if len(df_fondo) > 0:  # Solo si hay datos
            color = colores[i % len(colores)]
            plt.plot(df_fondo['fecha'], df_fondo['valor_unidad'], 
                    label=entidad, color=color, linewidth=2, alpha=0.8)
    
    plt.title(f'{titulo_base} - Comparación por Entidad', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor Unidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar gráfico
    Path("data/graficas_comparativas").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"data/graficas_comparativas/{nombre_archivo}.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    # 1. COMPARAR FONDO DE PENSIONES MODERADO entre entidades
fondos_moderado = {
    'Skandia': df_skandia_pensiones_moderado,
    'Protección': df_proteccion_pensiones_moderado,
    'Porvenir': df_porvenir_pensiones_moderado,
    'Colfondos': df_colfondos_pensiones_moderado
}

graficar_comparacion_entidades_por_fondo(
    fondos_moderado, 
    "Fondo de Pensiones Moderado",
    "comparacion_pensiones_moderado"
)

# 2. COMPARAR FONDO DE PENSIONES CONSERVADOR entre entidades
fondos_conservador = {
    'Skandia': df_skandia_pensiones_conservador,
    'Protección': df_proteccion_pensiones_conservador,
    'Porvenir': df_porvenir_pensiones_conservador,
    'Colfondos': df_colfondos_pensiones_conservador
}

graficar_comparacion_entidades_por_fondo(
    fondos_conservador, 
    "Fondo de Pensiones Conservador",
    "comparacion_pensiones_conservador"
)

# 3. COMPARAR CESANTÍAS LARGO PLAZO entre entidades
fondos_cesantias_largo = {
    'Skandia': df_skandia_cesantias_largo_plazo,
    'Protección': df_proteccion_cesantias_largo_plazo,
    'Porvenir': df_porvenir_cesantias_largo_plazo,
    'Colfondos': df_colfondos_cesantias_largo_plazo
}

graficar_comparacion_entidades_por_fondo(
    fondos_cesantias_largo, 
    "Fondo de Cesantías Largo Plazo",
    "comparacion_cesantias_largo"
)

# 4. COMPARAR CESANTÍAS CORTO PLAZO entre entidades
fondos_cesantias_corto = {
    'Skandia': df_skandia_cesantias_corto_plazo,
    'Protección': df_proteccion_cesantias_corto_plazo,
    'Porvenir': df_porvenir_cesantias_corto_plazo,
    'Colfondos': df_colfondos_cesantias_corto_plazo
}

graficar_comparacion_entidades_por_fondo(
    fondos_cesantias_corto, 
    "Fondo de Cesantías Corto Plazo",
    "comparacion_cesantias_corto"
)

# 5. COMPARAR FONDO MAYOR RIESGO entre entidades
fondos_mayor_riesgo = {
    'Skandia': df_skandia_pensiones_mayor_riesgo,
    'Protección': df_proteccion_pensiones_mayor_riesgo,
    'Porvenir': df_porvenir_pensiones_mayor_riesgo,
    'Colfondos': df_colfondos_pensiones_mayor_riesgo
}

graficar_comparacion_entidades_por_fondo(
    fondos_mayor_riesgo, 
    "Fondo de Pensiones Mayor Riesgo",
    "comparacion_pensiones_mayor_riesgo"
)
# 6. COMPARAR FONDO DE PENSIONES retiro programado entre entidades
fondos_retiro_programado = {
    'Skandia': df_skandia_pensiones_retiro_programado,
    'Protección': df_proteccion_pensiones_retiro_programado,
    'Porvenir': df_porvenir_pensiones_retiro_programado,
    'Colfondos': df_colfondos_pensiones_retiro_programado
}

graficar_comparacion_entidades_por_fondo(
    fondos_retiro_programado, 
    "Fondo de Pensiones retiro programado",
    "comparacion_pensiones_retiro_programado"
)

# 6. COMPARAR FONDO DE PENSIONES alternativo entre entidades

fondos_alternativo = {
    'Skandia': df_skandia_pensiones_alternativo,
    'Protección': df_proteccion_pensiones_alternativo,
    'Porvenir': df_porvenir_pensiones_alternativo,
    'Colfondos': df_colfondos_pensiones_alternativo
}

graficar_comparacion_entidades_por_fondo(
    fondos_alternativo, 
    "Fondo de Pensiones alternativo",
    "comparacion_pensiones_alternativo"
)

def evolucion_todos_fondos_entidad(entidad_nombre, dataframes_fondos):

    plt.figure(figsize=(16, 10))
    
    fondos_colores = {
        'Cesantías Largo Plazo': 'blue',
        'Cesantías Corto Plazo': 'lightblue',
        'Pensiones Moderado': 'green',
        'Pensiones Conservador': 'darkgreen', 
        'Pensiones Mayor Riesgo': 'red',
        'Pensiones Retiro Programado': 'orange',
        'Pensiones Alternativo': 'purple'
    }
    
    for fondo_nombre, color in fondos_colores.items():
        if fondo_nombre in dataframes_fondos and len(dataframes_fondos[fondo_nombre]) > 0:
            df = dataframes_fondos[fondo_nombre]
            # Normalizar para comparar (valores base 100)
            valor_base = df['valor_unidad'].iloc[0]
            df_normalizado = (df['valor_unidad'] / valor_base * 100)
            
            plt.plot(df['fecha'], df_normalizado, 
                    label=fondo_nombre, color=color, linewidth=2, alpha=0.7)
    
    plt.title(f'Evolución de Todos los Fondos - {entidad_nombre}\n(Valor de la Unidad Normalizado Base 100)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor Normalizado (Base 100)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'data/graficas_comparativas/evolucion_todos_fondos_{entidad_nombre.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def matriz_correlacion_fondos(entidad_nombre, dataframes_fondos):

    # Crear DataFrame con todos los fondos de la entidad
    datos_correlacion = {}
    
    for fondo_nombre, df in dataframes_fondos.items():
        if len(df) > 0:
            # Usar returns diarios para correlación
            df_temp = df.set_index('fecha')['valor_unidad'].sort_index()
            returns = df_temp.pct_change().dropna()
            datos_correlacion[fondo_nombre] = returns
    
    df_correlacion = pd.DataFrame(datos_correlacion)
    matriz_corr = df_correlacion.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))  # Máscara para triángulo superior
    sns.heatmap(matriz_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Coeficiente de Correlación'})
    plt.title(f'Matriz de Correlación - {entidad_nombre}\n(Returns Diarios)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'data/graficas_comparativas/correlacion_{entidad_nombre.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return matriz_corr

entidades_dataframes = {
    'Skandia': {
        'Cesantías Largo Plazo': df_skandia_cesantias_largo_plazo,
        'Cesantías Corto Plazo': df_skandia_cesantias_corto_plazo,
        'Pensiones Moderado': df_skandia_pensiones_moderado,
        'Pensiones Conservador': df_skandia_pensiones_conservador,
        'Pensiones Mayor Riesgo': df_skandia_pensiones_mayor_riesgo,
        'Pensiones Retiro Programado': df_skandia_pensiones_retiro_programado,
        'Pensiones Alternativo': df_skandia_pensiones_alternativo
    },
    'Protección': {
        'Cesantías Largo Plazo': df_proteccion_cesantias_largo_plazo,
        'Cesantías Corto Plazo': df_proteccion_cesantias_corto_plazo,
        'Pensiones Moderado': df_proteccion_pensiones_moderado,   
        'Pensiones Conservador': df_proteccion_pensiones_conservador,
        'Pensiones Mayor Riesgo': df_proteccion_pensiones_mayor_riesgo,
        'Pensiones Retiro Programado': df_proteccion_pensiones_retiro_programado,
        'Pensiones Alternativo': df_proteccion_pensiones_alternativo
    },
    'Porvenir': {
        'Cesantías Largo Plazo': df_porvenir_cesantias_largo_plazo,
        'Cesantías Corto Plazo': df_porvenir_cesantias_corto_plazo,
        'Pensiones Moderado': df_porvenir_pensiones_moderado,   
        'Pensiones Conservador': df_porvenir_pensiones_conservador,
        'Pensiones Mayor Riesgo': df_porvenir_pensiones_mayor_riesgo,
        'Pensiones Retiro Programado': df_porvenir_pensiones_retiro_programado,
        'Pensiones Alternativo': df_porvenir_pensiones_alternativo
    },
    'Colfondos': {
        'Cesantías Largo Plazo': df_colfondos_cesantias_largo_plazo,
        'Cesantías Corto Plazo': df_colfondos_cesantias_corto_plazo,
        'Pensiones Moderado': df_colfondos_pensiones_moderado,   
        'Pensiones Conservador': df_colfondos_pensiones_conservador,
        'Pensiones Mayor Riesgo': df_colfondos_pensiones_mayor_riesgo,
        'Pensiones Retiro Programado': df_colfondos_pensiones_retiro_programado,
        'Pensiones Alternativo': df_colfondos_pensiones_alternativo       
    }
}
evolucion_todos_fondos_entidad('Skandia', entidades_dataframes['Skandia'])
evolucion_todos_fondos_entidad('Protección', entidades_dataframes['Protección'])
evolucion_todos_fondos_entidad('Porvenir', entidades_dataframes['Porvenir'])
evolucion_todos_fondos_entidad('Colfondos', entidades_dataframes['Colfondos'])  
matriz_correlacion_fondos('Skandia', entidades_dataframes['Skandia'])
matriz_correlacion_fondos('Protección', entidades_dataframes['Protección'])
matriz_correlacion_fondos('Porvenir', entidades_dataframes['Porvenir'])
matriz_correlacion_fondos('Colfondos', entidades_dataframes['Colfondos'])

# Creamos los Lags: pasos hacia atras que estas comparando, creamos lag para todos los dias

for lag in range(1, len(df)//30, 30):  # Lags cada 30 días hasta un año
    df[f'lag_{lag}'] = df['valor_unidad'].shift(lag) 


# =============================================================================
# ANÁLISIS EXPLORATORIO (EDA)

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO COMPLETO (EDA)")
print("="*60)

# Asegurar que el directorio para gráficas existe
Path("data/graficas_comparativas").mkdir(parents=True, exist_ok=True)

# 1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
print("\n1. ESTADÍSTICAS DESCRIPTIVAS:")
print(df[['valor_unidad']].describe())

# Estadísticas por tipo de fondo
print("\nEstadísticas por tipo de fondo:")
print(print(df.groupby('tipo_fondo', observed=True)['valor_unidad'].describe()))

# 2. ANÁLISIS DE DISTRIBUCIÓN
print("\n2. ANÁLISIS DE DISTRIBUCIÓN:")

# Histograma y densidad
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
df['valor_unidad'].hist(bins=50, alpha=0.7)
plt.title('Distribución de valor_unidad')
plt.xlabel('Valor Unidad')
plt.ylabel('Frecuencia')
plt.savefig('data/graficas_comparativas/distribucion_valor_unidad.png', dpi=300, bbox_inches='tight')

plt.subplot(1, 2, 2)
df['valor_unidad'].plot(kind='density')
plt.title('Densidad de valor_unidad')
plt.xlabel('Valor Unidad')

plt.tight_layout()
plt.savefig('data/graficas_comparativas/densidad_valor_unidad.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: distribucion_valor_unidad.png y densidad_valor_unidad.png")

# 3. ANÁLISIS TEMPORAL AVANZADO
print("\n3. ANÁLISIS TEMPORAL:")

# Evolución temporal promedio por año
evolucion_anual = df.groupby('año')['valor_unidad'].agg(['mean', 'std', 'min', 'max'])
print("Evolución anual:")
print(evolucion_anual)

plt.figure(figsize=(12, 6))
evolucion_anual['mean'].plot(kind='line', marker='o')
plt.fill_between(evolucion_anual.index, 
                 evolucion_anual['mean'] - evolucion_anual['std'], 
                 evolucion_anual['mean'] + evolucion_anual['std'], 
                 alpha=0.2)
plt.title('Evolución Anual del Valor Unidad (media ± desviación)')
plt.xlabel('Año')
plt.ylabel('Valor Unidad')
plt.grid(True, alpha=0.3)
plt.savefig('data/graficas_comparativas/evolucion_anual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: evolucion_anual.png")

# 4. ANÁLISIS DE CORRELACIONES TEMPORALES (AUTOCORRELACIÓN)
print("\n4. ANÁLISIS DE AUTOCORRELACIÓN:")

# Ejemplo con un fondo específico para análisis de autocorrelación
fondo_ejemplo = df_skandia_pensiones_moderado.set_index('fecha')['valor_unidad']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plot_acf(fondo_ejemplo, lags=30, ax=plt.gca())
plt.title('Autocorrelación (Fondo Moderado Skandia)')

plt.subplot(1, 2, 2)
plot_pacf(fondo_ejemplo, lags=30, ax=plt.gca())
plt.title('Autocorrelación Parcial (Fondo Moderado Skandia)')

plt.tight_layout()
plt.savefig('data/graficas_comparativas/autocorrelacion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: autocorrelacion.png")

# 5. DESCOMPOSICIÓN ESTACIONAL
print("\n5. DESCOMPOSICIÓN ESTACIONAL:")

# Preparar datos mensuales para descomposición
fondo_mensual = fondo_ejemplo.resample('ME').mean()  # 'ME' en lugar de 'M'

# Descomposición estacional
try:
    descomposicion = seasonal_decompose(fondo_mensual, model='additive', period=12)
    
    fig = descomposicion.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Descomposición Estacional - Fondo Moderado Skandia (Mensual)', fontsize=14)
    plt.savefig('data/graficas_comparativas/descomposicion_estacional.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Descomposición estacional completada")
    print("✓ Gráfica guardada: descomposicion_estacional.png")
except Exception as e:
    print(f"✗ Error en descomposición estacional: {e}")

# 6. ANÁLISIS DE ESTACIONALIDAD POR MES
print("\n6. ANÁLISIS DE ESTACIONALIDAD POR MES:")

# Gráfico de estacionalidad por mes
estacionalidad_mensual = df.groupby('mes')['valor_unidad'].mean()

plt.figure(figsize=(10, 6))
estacionalidad_mensual.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('Comportamiento Estacional Promedio por Mes')
plt.xlabel('Mes')
plt.ylabel('Valor Unidad Promedio')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.savefig('data/graficas_comparativas/estacionalidad_mensual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: estacionalidad_mensual.png")

# 7. ANÁLISIS DE OUTLIERS POR CATEGORÍA
print("\n7. ANÁLISIS DE OUTLIERS POR CATEGORÍA:")

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='tipo_fondo', y='valor_unidad')
plt.title('Distribución y Outliers por Tipo de Fondo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/graficas_comparativas/boxplot_tipos_fondo.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: boxplot_tipos_fondo.png")

# 8. ANÁLISIS DE VOLATILIDAD (ROLLING STATISTICS)
print("\n8. ANÁLISIS DE VOLATILIDAD:")

# Calcular volatilidad rolling (ventana de 30 días)
fondo_ejemplo_vol = fondo_ejemplo.rolling(window=30).std()

plt.figure(figsize=(12, 6))
plt.plot(fondo_ejemplo_vol.index, fondo_ejemplo_vol.values, color='red', alpha=0.7)
plt.title('Volatilidad Rolling (30 días) - Fondo Moderado Skandia')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad (Desviación Estándar)')
plt.grid(True, alpha=0.3)
plt.savefig('data/graficas_comparativas/volatilidad_rolling.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: volatilidad_rolling.png")

# 9. MATRIZ DE CORRELACIÓN ENTRE TIPOS DE FONDO
print("\n9. MATRIZ DE CORRELACIÓN ENTRE TIPOS DE FONDO:")

# Crear pivot table para correlación entre tipos de fondo
pivot_corr = df.pivot_table(
    index='fecha', 
    columns='tipo_fondo', 
    values='valor_unidad',
    observed=False  # o observed=True según tu necesidad
).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlación entre Diferentes Tipos de Fondos')
plt.tight_layout()
plt.savefig('data/graficas_comparativas/correlacion_tipos_fondo.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: correlacion_tipos_fondo.png")

# 10. ANÁLISIS DE TENDENCIAS (ROLLING MEAN)
print("\n10. ANÁLISIS DE TENDENCIAS:")

# Tendencia con media móvil
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
plt.savefig('data/graficas_comparativas/analisis_tendencia.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: analisis_tendencia.png")

# 11. ANÁLISIS DE OUTLIERS DETALLADO
print("\n11. ANÁLISIS DETALLADO DE OUTLIERS:")

# Gráfico de dispersión para identificar outliers visualmente
plt.figure(figsize=(12, 6))
plt.scatter(df.index, df['valor_unidad'], c=df['es_outlier'], cmap='coolwarm', alpha=0.6)
plt.title('Identificación Visual de Outliers')
plt.xlabel('Índice')
plt.ylabel('Valor Unidad')
plt.savefig('data/graficas_comparativas/outliers_detallado.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: outliers_detallado.png")

# 12. ANÁLISIS DE DENSIDAD POR TIPO DE FONDO
print("\n12. ANÁLISIS DE DENSIDAD POR TIPO DE FONDO:")

plt.figure(figsize=(12, 6))
for tipo in df['tipo_fondo'].unique():
    subset = df[df['tipo_fondo'] == tipo]
    plt.hist(subset['valor_unidad'], bins=50, alpha=0.5, label=tipo, density=True)
plt.title('Distribución de Densidad por Tipo de Fondo')
plt.xlabel('Valor Unidad')
plt.ylabel('Densidad')
plt.legend()
plt.savefig('data/graficas_comparativas/densidad_tipos_fondo.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: densidad_tipos_fondo.png")

# 13. ANÁLISIS TEMPORAL POR AÑO
print("13. ANÁLISIS TEMPORAL POR AÑO:")
years = sorted(df['año'].unique())
n_years = len(years)

# Calcular dimensiones dinámicas para la grid
n_cols = 3
n_rows = (n_years + n_cols - 1) // n_cols  # División entera hacia arriba

Path("images/analisis_exploratorio").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(15, 5 * n_rows))

for i, year in enumerate(years, 1):
    plt.subplot(n_rows, n_cols, i)
    
    data_year = df[df['año'] == year]
    
    # Graficar cada tipo de fondo
    for fondo in data_year['tipo_fondo'].unique():
        data_fondo = data_year[data_year['tipo_fondo'] == fondo]
        monthly_avg = data_fondo.groupby('mes')['valor_unidad'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, label=fondo, marker='o')
    
    plt.title(f'Evolución mensual {year}')
    plt.xlabel('Mes')
    plt.ylabel('Valor Unitario Promedio')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    
    # Solo mostrar leyenda en algunos gráficos para evitar sobrecarga
    if i == 1:  # Solo en el primer gráfico
        plt.legend()

plt.tight_layout()
plt.savefig('images/analisis_exploratorio/evolucion_mensual_por_año.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfica guardada: evolucion_mensual_por_año.png")

# 14. RESUMEN ESTADÍSTICO PARA MODELADO
print("\n14. RESUMEN PARA MODELADO:")

print(f"• Rango temporal: {df['fecha'].min()} to {df['fecha'].max()}")
print(f"• Total de observaciones: {len(df):,}")
print(f"• Número de entidades únicas: {df['nombre_entidad'].nunique()}")
print(f"• Número de fondos únicos: {df['nombre_fondo'].nunique()}")
print(f"• Rango de valores: {df['valor_unidad'].min():.2f} - {df['valor_unidad'].max():.2f}")
print(f"• Coeficiente de variación: {(df['valor_unidad'].std() / df['valor_unidad'].mean() * 100):.2f}%")

# 15. EXPORTAR DATASET LISTO PARA MODELADO
print("\n15. EXPORTANDO DATASET PARA MODELADO...")

# Dataset limpio y enriquecido listo para modelos
df_modelado = df.copy()

# Asegurar que no hay valores nulos en variables clave
df_modelado = df_modelado.dropna(subset=['valor_unidad', 'fecha', 'nombre_entidad', 'nombre_fondo'])

# Exportar dataset listo para modelado
df_modelado.to_csv("data/processed/pensiones_listo_modelado.csv", index=False, encoding='utf-8')
print("✓ Dataset listo para modelado exportado a: data/processed/pensiones_listo_modelado.csv")

# 16. CREACIÓN OPTIMIZADA DE LAGS (CORRECCIÓN DEL ERROR)
print("\n16. CREACIÓN OPTIMIZADA DE VARIABLES TEMPORALES:")

# Versión optimizada para evitar fragmentación
try:
    # En lugar de múltiples inserts, usa concat
    lags = [1, 7, 30, 90, 180, 365]
    lag_columns = {}

    for lag in lags:
        lag_columns[f'lag_{lag}'] = df['valor_unidad'].shift(lag)

    lag_df = pd.DataFrame(lag_columns)
    df = pd.concat([df, lag_df], axis=1)
         
except Exception as e:
    print(f"✗ Error creando lags: {e}")

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO COMPLETADO ✓")
print(f"Total de gráficas generadas: 13")
print("Todas las gráficas guardadas en: data/graficas_comparativas/")
print("="*60)

# =============================================================================
# MODELADO DE SERIES TEMPORALES - EXTENSIÓN DEL PIPELINE
# =============================================================================

print("\n" + "="*60)
print("FASE 4: MODELADO DE SERIES TEMPORALES")
print("="*60)

# Importaciones adicionales para modelado
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def analizar_estacionariedad(serie, nombre_serie=""):
    """
    Analiza la estacionariedad de una serie temporal usando la prueba ADF
    
    Parameters:
    serie (pd.Series): Serie temporal a analizar
    nombre_serie (str): Nombre de la serie para reporte
    
    Returns:
    dict: Resultados de la prueba de estacionariedad
    """
    print(f"\n--- Análisis de Estacionariedad: {nombre_serie} ---")
    
    # Prueba de Dickey-Fuller Aumentada
    resultado_adf = adfuller(serie.dropna())
    
    metricas = {
        'estadistico_adf': resultado_adf[0],
        'p_valor': resultado_adf[1],
        'valores_criticos': resultado_adf[4],
        'es_estacionaria': resultado_adf[1] < 0.05
    }
    
    print(f"Estadístico ADF: {metricas['estadistico_adf']:.4f}")
    print(f"P-valor: {metricas['p_valor']:.4f}")
    
    if metricas['es_estacionaria']:
        print("✓ La serie ES estacionaria (p-valor < 0.05)")
    else:
        print("✗ La serie NO es estacionaria (p-valor > 0.05)")
        print("  Se requiere diferenciación para modelado ARIMA")
    
    return metricas

def entrenar_modelo_arima(serie, orden, nombre_serie=""):
    """
    Entrena un modelo ARIMA para una serie temporal
    
    Parameters:
    serie (pd.Series): Serie temporal estacionaria
    orden (tuple): Orden (p,d,q) del modelo ARIMA
    nombre_serie (str): Nombre de la serie
    
    Returns:
    tuple: Modelo entrenado y métricas
    """
    print(f"Entrenando ARIMA{orden} para {nombre_serie}...")
    
    try:
        modelo = ARIMA(serie, order=orden)
        modelo_ajustado = modelo.fit()
        
        metricas = {
            'aic': modelo_ajustado.aic,
            'bic': modelo_ajustado.bic,
            'residuos_media': modelo_ajustado.resid.mean(),
            'residuos_std': modelo_ajustado.resid.std()
        }
        
        print(f"✓ ARIMA{orden} entrenado exitosamente")
        print(f"  AIC: {metricas['aic']:.2f}, BIC: {metricas['bic']:.2f}")
        
        return modelo_ajustado, metricas
        
    except Exception as e:
        print(f"✗ Error entrenando ARIMA{orden}: {e}")
        return None, None

def buscar_mejor_arima(serie, parametros_a_probar, nombre_serie=""):
    """
    Realiza búsqueda en grid para encontrar los mejores parámetros ARIMA
    
    Parameters:
    serie (pd.Series): Serie temporal
    parametros_a_probar (list): Lista de tuplas (p,d,q) a probar
    nombre_serie (str): Nombre de la serie
    
    Returns:
    dict: Mejor modelo y métricas
    """
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
        print(f"\n🎯 MEJOR MODELO ENCONTRADO:")
        print(f"   ARIMA{mejor_orden} con AIC: {mejores_metricas['aic']:.2f}")
        
        return {
            'modelo': mejor_modelo,
            'orden': mejor_orden,
            'metricas': mejores_metricas
        }
    else:
        print("✗ No se pudo encontrar un modelo adecuado")
        return None

def evaluar_pronostico(real, pronosticado, nombre_serie=""):
    """
    Evalúa la calidad de los pronósticos del modelo
    
    Parameters:
    real (array): Valores reales
    pronosticado (array): Valores pronosticados
    nombre_serie (str): Nombre de la serie
    
    Returns:
    dict: Métricas de evaluación
    """
    print(f"\n--- Evaluación de Pronósticos: {nombre_serie} ---")
    
    mse = np.mean((real - pronosticado)**2)
    mae = np.mean(np.abs(real - pronosticado))
    mape = np.mean(np.abs((real - pronosticado) / real)) * 100
    rmse = np.sqrt(mse)
    
    metricas = {
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse
    }
    
    print("Métricas de evaluación:")
    for metrica, valor in metricas.items():
        print(f"  {metrica}: {valor:.4f}")
    
    print(f"  Error porcentual promedio (MAPE): {mape:.2f}%")
    
    return metricas

def pipeline_modelado_completo(df_serie, nombre_serie, columna_valor='valor_unidad'):
    """
    Pipeline completo para modelado de series temporales
    
    Parameters:
    df_serie (pd.DataFrame): DataFrame con la serie temporal
    nombre_serie (str): Nombre identificador de la serie
    columna_valor (str): Columna con los valores a modelar
    
    Returns:
    dict: Resultados completos del modelado
    """
    print(f"\n{'='*50}")
    print(f"INICIANDO PIPELINE DE MODELADO: {nombre_serie}")
    print(f"{'='*50}")
    
    resultados = {}
    
    # 1. Preparación de datos
    print("\n1. 📊 PREPARANDO DATOS...")
    serie = df_serie.set_index('fecha')[columna_valor].sort_index()
    resultados['serie_original'] = serie.copy()
    
    # 2. Análisis de estacionariedad
    print("\n2. 🔍 ANALIZANDO ESTACIONARIEDAD...")
    resultados['estacionariedad'] = analizar_estacionariedad(serie, nombre_serie)
    
    # 3. Diferenciación si no es estacionaria
    if not resultados['estacionariedad']['es_estacionaria']:
        print("\n3. ⚙️  APLICANDO DIFERENCIACIÓN...")
        serie_diff = serie.diff().dropna()
        resultados['serie_diferenciada'] = serie_diff
        resultados['estacionariedad_diff'] = analizar_estacionariedad(serie_diff, f"{nombre_serie} (diferenciada)")
        serie_para_modelar = serie_diff
    else:
        serie_para_modelar = serie
    
    # 4. Búsqueda del mejor modelo ARIMA
    print("\n4. 🤖 BUSCANDO MEJOR MODELO ARIMA...")
    parametros_a_probar = [(1,0,0), (1,1,1), (2,1,2), (0,1,1), (1,1,0)]
    resultados['mejor_modelo'] = buscar_mejor_arima(serie_para_modelar, parametros_a_probar, nombre_serie)
    
    if resultados['mejor_modelo']:
        modelo = resultados['mejor_modelo']['modelo']
        
        # 5. Validación del modelo
        print("\n5. ✅ VALIDANDO MODELO...")
        
        # Dividir en train/test (80/20)
        train_size = int(len(serie_para_modelar) * 0.8)
        train, test = serie_para_modelar[:train_size], serie_para_modelar[train_size:]
        
        # Entrenar modelo con datos de entrenamiento
        modelo_train = ARIMA(train, order=resultados['mejor_modelo']['orden']).fit()
        
        # Pronosticar
        pronostico = modelo_train.forecast(steps=len(test))
        
        # Evaluar pronóstico
        resultados['evaluacion'] = evaluar_pronostico(test.values, pronostico.values, nombre_serie)
        
        # 6. Pronóstico futuro
        print("\n6. 🔮 GENERANDO PRONÓSTICOS FUTUROS...")
        pronostico_futuro = modelo.forecast(steps=30)  # 30 días futuros
        resultados['pronostico_futuro'] = pronostico_futuro
        
        print(f"📈 Pronóstico para los próximos 30 días:")
        print(f"   Valor inicial: {serie_para_modelar.iloc[-1]:.2f}")
        print(f"   Tendencia pronosticada: {'↗️ Alza' if pronostico_futuro.iloc[-1] > serie_para_modelar.iloc[-1] else '↘️ Baja'}")
    
    # 7. Guardar resultados
    print("\n7. 💾 GUARDANDO RESULTADOS...")
    
    # Crear directorio para resultados de modelado
    Path("data/modelos").mkdir(parents=True, exist_ok=True)
    
    # Exportar resumen del modelado
    resumen_modelado = {
        'serie': nombre_serie,
        'mejor_modelo': f"ARIMA{resultados.get('mejor_modelo', {}).get('orden', 'N/A')}",
        'aic': resultados.get('mejor_modelo', {}).get('metricas', {}).get('aic', 'N/A'),
        'estacionaria': resultados.get('estacionariedad', {}).get('es_estacionaria', False),
        'mape': resultados.get('evaluacion', {}).get('MAPE', 'N/A')
    }
    
    # Guardar resumen
    pd.Series(resumen_modelado).to_csv(f"data/modelos/resumen_{nombre_serie.replace(' ', '_').lower()}.csv")
    
    print(f"✓ Pipeline de modelado completado para {nombre_serie}")
    
    return resultados

# =============================================================================
# EJECUCIÓN DEL MODELADO EN SERIES SELECCIONADAS
# =============================================================================

print("\n" + "="*60)
print("EJECUTANDO MODELADO EN SERIES REPRESENTATIVAS")
print("="*60)

# Seleccionar series representativas para modelado
series_a_modelar = {
    "Fondo Moderado Skandia": df_skandia_pensiones_moderado,
    "Fondo Conservador Porvenir": df_porvenir_pensiones_conservador,
    "Cesantías Largo Plazo Colfondos": df_colfondos_cesantias_largo_plazo
}

resultados_modelado = {}

for nombre_serie, df_serie in series_a_modelar.items():
    if len(df_serie) > 100:  # Solo modelar series con suficiente data
        try:
            resultados = pipeline_modelado_completo(df_serie, nombre_serie)
            resultados_modelado[nombre_serie] = resultados
        except Exception as e:
            print(f"✗ Error en modelado de {nombre_serie}: {e}")
    else:
        print(f"⚠️  Serie {nombre_serie} muy corta para modelado ({len(df_serie)} registros)")

# =============================================================================
# ANÁLISIS COMPARATIVO DE MODELOS
# =============================================================================

print("\n" + "="*60)
print("ANÁLISIS COMPARATIVO DE MODELOS")
print("="*60)

if resultados_modelado:
    # Crear DataFrame comparativo
    comparacion_modelos = []
    
    for nombre, resultados in resultados_modelado.items():
        if resultados.get('mejor_modelo'):
            comparacion_modelos.append({
                'Serie': nombre,
                'Mejor Modelo': f"ARIMA{resultados['mejor_modelo']['orden']}",
                'AIC': resultados['mejor_modelo']['metricas']['aic'],
                'Estacionaria': resultados['estacionariedad']['es_estacionaria'],
                'MAPE (%)': resultados.get('evaluacion', {}).get('MAPE', 'N/A')
            })
    
    df_comparacion = pd.DataFrame(comparacion_modelos)
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print(df_comparacion.to_string(index=False))
    
    # Guardar comparación
    df_comparacion.to_csv("data/modelos/comparacion_modelos.csv", index=False)
    print("✓ Comparación de modelos guardada en: data/modelos/comparacion_modelos.csv")
    
    # Análisis de resultados
    print("\n🔍 INTERPRETACIÓN DE RESULTADOS:")
    
    mejor_modelo = df_comparacion.loc[df_comparacion['AIC'].idxmin()]
    print(f"• Mejor modelo general: {mejor_modelo['Serie']} ({mejor_modelo['Mejor Modelo']})")
    print(f"• AIC más bajo: {mejor_modelo['AIC']:.2f}")
    
    # Recomendaciones basadas en resultados
    print("\n💡 RECOMENDACIONES PARA PRÓXIMOS PASOS:")
    print("1. Para series estacionarias: Considerar modelos ARMA puros")
    print("2. Para series no estacionarias: Explorar diferenciación estacional (SARIMA)")
    print("3. Series con alto MAPE: Investigar outliers y eventos atípicos")
    print("4. Considerar modelos de machine learning (Random Forest, XGBoost) para comparación")

else:
    print("✗ No se pudieron generar modelos para comparación")

# =============================================================================
# VISUALIZACIÓN DE RESULTADOS DEL MODELADO
# =============================================================================

def visualizar_resultados_modelado(resultados_modelado):
    """
    Genera visualizaciones para los resultados del modelado
    """
    print("\n🎨 GENERANDO VISUALIZACIONES DE RESULTADOS...")
    
    Path("data/graficas_modelado").mkdir(parents=True, exist_ok=True)
    
    for nombre_serie, resultados in resultados_modelado.items():
        if resultados.get('mejor_modelo'):
            try:
                # Gráfico de serie original vs ajustada
                plt.figure(figsize=(15, 10))
                
                # Subplot 1: Serie original y ajustada
                plt.subplot(2, 2, 1)
                serie_original = resultados['serie_original']
                modelo = resultados['mejor_modelo']['modelo']
                
                plt.plot(serie_original.index, serie_original.values, label='Original', alpha=0.7)
                plt.plot(modelo.fittedvalues.index, modelo.fittedvalues, label='Ajustado', alpha=0.8)
                plt.title(f'Serie Original vs Ajustada\n{nombre_serie}')
                plt.legend()
                plt.xticks(rotation=45)
                
                # Subplot 2: Residuos
                plt.subplot(2, 2, 2)
                residuos = modelo.resid
                plt.plot(residuos.index, residuos.values)
                plt.title('Residuos del Modelo')
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xticks(rotation=45)
                
                # Subplot 3: Distribución de residuos
                plt.subplot(2, 2, 3)
                plt.hist(residuos.dropna(), bins=50, alpha=0.7, density=True)
                plt.title('Distribución de Residuos')
                plt.xlabel('Residuos')
                plt.ylabel('Densidad')
                
                # Subplot 4: Pronóstico si está disponible
                plt.subplot(2, 2, 4)
                if 'pronostico_futuro' in resultados:
                    pronostico = resultados['pronostico_futuro']
                    ultimos_30 = serie_original.tail(30)
                    
                    plt.plot(ultimos_30.index, ultimos_30.values, label='Últimos 30 días', color='blue')
                    plt.plot(pronostico.index, pronostico.values, label='Pronóstico 30 días', color='red', linestyle='--')
                    plt.title('Pronóstico a 30 días')
                    plt.legend()
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f"data/graficas_modelado/resultados_{nombre_serie.replace(' ', '_').lower()}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Gráficas guardadas para: {nombre_serie}")
                
            except Exception as e:
                print(f"✗ Error generando gráficas para {nombre_serie}: {e}")

# Ejecutar visualizaciones
visualizar_resultados_modelado(resultados_modelado)

print("\n" + "="*60)
print("PIPELINE DE MODELADO COMPLETADO ✓")
print("="*60)
print("RESUMEN EJECUTADO:")
print(f"• Series modeladas: {len(resultados_modelado)}")
print(f"• Modelos generados: {sum(1 for r in resultados_modelado.values() if r.get('mejor_modelo'))}")
print(f"• Resultados guardados en: data/modelos/")
print(f"• Gráficas guardadas en: data/graficas_modelado/")
print("="*60)

# ============================================================
# REGRESIÓN LINEAL Y GRÁFICA COMPARATIVA
#   Serie: Skandia - Cesantías Corto Plazo
#   Variables: fecha (X) -> valor_unidad (y)
# ============================================================

from sklearn.linear_model import LinearRegression

# 1) Tomar y ordenar la serie de interés
serie_cp = (
    df_skandia_cesantias_corto_plazo[['fecha', 'valor_unidad']]
    .dropna()
    .sort_values('fecha')
    .copy()
)
assert len(serie_cp) > 0, "No hay datos para Skandia Cesantías Corto Plazo."

# 2) Convertir fechas a días desde el inicio (variable numérica X)
t0 = serie_cp['fecha'].min()
X = (serie_cp['fecha'] - t0).dt.days.values.reshape(-1, 1)  # (n,1)
y = serie_cp['valor_unidad'].values

# 3) Ajustar la regresión lineal
lin = LinearRegression()
lin.fit(X, y)

# 4) Predicción sobre todo el rango observado (para la línea ajustada sobre la muestra)
y_fit = lin.predict(X)

# 5) Eje temporal para dibujar la recta desde 2016 hasta 2026 (o hasta fin de datos)
inicio_linea = pd.Timestamp('2016-01-01')
fin_datos     = serie_cp['fecha'].max()
fin_linea     = max(fin_datos, pd.Timestamp('2026-12-31'))  # extiende hasta 2026 si hace falta
fechas_linea  = pd.date_range(start=inicio_linea, end=fin_linea, freq='D')

# A estas fechas “globales” también les calculamos X en días desde t0
X_linea = ((fechas_linea - t0).days.values).reshape(-1, 1)
y_linea = lin.predict(X_linea)

# 6) Gráfica: serie real + recta de regresión (2016–2026)
plt.figure(figsize=(14, 6))
plt.plot(serie_cp['fecha'], serie_cp['valor_unidad'], label='Real (Skandia Cesantías Corto Plazo)', alpha=0.85)
plt.plot(fechas_linea, y_linea, '--', linewidth=2.2, label='Recta de regresión (2016–2026)')

plt.title('Serie real vs. Recta de regresión lineal\nSkandia – Cesantías Corto Plazo')
plt.xlabel('Fecha')
plt.ylabel('Valor Unidad')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('data/graficas_comparativas/reg_lineal_skandia_cesantias_corto_plazo.png', dpi=300, bbox_inches='tight')
plt.show()

# (Opcional) Métricas rápidas sobre la muestra
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae  = mean_absolute_error(y, y_fit)
rmse = np.sqrt(mean_squared_error(y, y_fit))
mape = np.mean(np.abs((y - y_fit) / y)) * 100
print("Métricas (sobre la muestra): MAE={:.2f}  RMSE={:.2f}  MAPE={:.2f}%".format(mae, rmse, mape))
# ============================================================
# REGRESIÓN LINEAL: PRONÓSTICO A 1 AÑO PARA TODOS LOS FONDOS DE SKANDIA
# ============================================================

fondos_skandia = {
    "Cesantías Corto Plazo": df_skandia_cesantias_corto_plazo,
    "Cesantías Largo Plazo": df_skandia_cesantias_largo_plazo,
    "Pensiones Moderado": df_skandia_pensiones_moderado,
    "Pensiones Conservador": df_skandia_pensiones_conservador,
    "Pensiones Mayor Riesgo": df_skandia_pensiones_mayor_riesgo,
    "Pensiones Retiro Programado": df_skandia_pensiones_retiro_programado,
    "Pensiones Alternativo": df_skandia_pensiones_alternativo
}

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

resultados_skandia = {}

for nombre_fondo, df_fondo in fondos_skandia.items():
    if len(df_fondo) < 30:
        print(f"⚠️ Fondo '{nombre_fondo}' tiene muy pocos datos. Se omite.")
        continue

    # 1️⃣ Preparar la serie
    df_temp = df_fondo[['fecha', 'valor_unidad']].dropna().sort_values('fecha').copy()
    t0 = df_temp['fecha'].min()
    X = (df_temp['fecha'] - t0).dt.days.values.reshape(-1, 1)
    y = df_temp['valor_unidad'].values

    # 2️⃣ Entrenar regresión lineal
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_fit = modelo.predict(X)

    # 3️⃣ Pronóstico a 1 año desde la última fecha
    ultima_fecha = df_temp['fecha'].max()
    fecha_futura = ultima_fecha + pd.Timedelta(days=365)
    dias_futuros = np.array([(fecha_futura - t0).days]).reshape(-1, 1)
    pred_futuro = modelo.predict(dias_futuros)[0]

    # 4️⃣ Crear recta extendida (de inicio a +1 año)
    fechas_linea = pd.date_range(start=t0, end=fecha_futura, freq='D')
    X_linea = ((fechas_linea - t0).days.values).reshape(-1, 1)
    y_linea = modelo.predict(X_linea)

    # 5️⃣ Métricas básicas
    mae = mean_absolute_error(y, y_fit)
    rmse = np.sqrt(mean_squared_error(y, y_fit))
    mape = np.mean(np.abs((y - y_fit) / y)) * 100

    resultados_skandia[nombre_fondo] = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Valor_predicho_1año": pred_futuro,
        "Última_fecha": ultima_fecha.strftime('%Y-%m-%d'),
        "Fecha_predicha": fecha_futura.strftime('%Y-%m-%d')
    }

    # 6️⃣ Gráfica comparativa
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
    plt.savefig(f"data/graficas_modelado/reg_lineal_skandia_{nombre_fondo.replace(' ', '_').lower()}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 📊 RESUMEN DE PRONÓSTICOS
# ============================================================
df_resultados_skandia = pd.DataFrame(resultados_skandia).T
df_resultados_skandia.to_csv("data/modelos/predicciones_skandia_1año.csv", encoding='utf-8')

print("\n📈 PRONÓSTICOS A 1 AÑO (Regresión Lineal - Skandia):")
print(df_resultados_skandia.round(2))
print("\n✓ Resultados guardados en: data/modelos/predicciones_skandia_1año.csv")
print("✓ Gráficas guardadas en: data/graficas_modelado/")

# ===== IPC Colombia (Banco Mundial) vs Skandia Moderado (base 100) =====
import re

# 1) Cargar CSV del BM (saltando metadatos)
ipc_raw = pd.read_csv(
    "/home/wolfang-sanchez/Documents/proyecto/data/API_FP.CPI.TOTL_DS2_es_csv_v2_59981.csv",
    skiprows=4
)

# 2) Columna de país (según idioma del archivo)
col_country = next(c for c in ipc_raw.columns
                   if c.lower().startswith(("country name", "nombre del país", "nombre del pais")))

# 3) Filtrar Colombia
ipc_col = ipc_raw.loc[ipc_raw[col_country].str.lower() == "colombia"].copy()

# 4) Detectar columnas de años (2016, "2016 [YR2016]", etc.) y renombrarlas a int
def es_col_anio(c): return re.match(r"^\d{4}($|\s|\[)", str(c)) is not None
year_cols = [c for c in ipc_col.columns if es_col_anio(c)]
rename_map = {c: int(re.match(r"^(\d{4})", str(c)).group(1)) for c in year_cols}
ipc_col = ipc_col[year_cols].rename(columns=rename_map)

# 5) Asegurar numérico y quedarnos con 2016..último disponible (sin provocar KeyError)
ipc_col = ipc_col.apply(pd.to_numeric, errors="coerce")
anio_ini, anio_fin_deseado = 2016, 2025
anios_disp = sorted(ipc_col.columns.dropna())
if not anios_disp:
    raise ValueError("No se detectaron columnas de años numéricos en el CSV del IPC.")

anio_fin = min(anio_fin_deseado, anios_disp[-1])
anios_sel = [a for a in anios_disp if anio_ini <= a <= anio_fin]

# 6) Serie IPC anual (fin de año)
ipc_series = ipc_col.loc[:, anios_sel].T.squeeze()
ipc_series.index = pd.to_datetime(ipc_series.index.astype(str) + "-12-31")
ipc_series.name = "IPC Colombia (Base 100)"

# 7) Skandia Moderado → base 100 y anualizar a fin de año
sk_mod = df_skandia_pensiones_moderado.set_index("fecha")["valor_unidad"].sort_index()

def base100(s):
    s = s.dropna()
    return s / s.iloc[0] * 100

# Normalizamos desde 2016
sk_b100_diario = base100(sk_mod[sk_mod.index >= "2016-01-01"])
sk_b100_anual  = sk_b100_diario.resample("A-DEC").last()
ipc_b100       = base100(ipc_series[ipc_series.index >= "2016-01-01"])

# 8) Alinear y revisar
cmp = pd.concat([ipc_b100.rename("IPC (base=100)"),
                 sk_b100_anual.rename("Skandia Moderado (base=100)")], axis=1)

# Si hay NaN (p.ej. por años sin dato), los interpolamos suavemente para que se vea la línea
cmp = cmp.sort_index().interpolate(limit_direction="both")

# 9) Graficar con marcadores para el IPC (se ve sí o sí)
plt.figure(figsize=(14, 6))
plt.plot(cmp.index, cmp["Skandia Moderado (base=100)"], label="Fondo Skandia Moderado (Base 100)", linewidth=2)
plt.plot(cmp.index, cmp["IPC (base=100)"], "--", label="IPC Colombia (Base 100)", linewidth=2, marker="o")

plt.title("Comparación: Evolución del IPC vs. Fondo de Pensiones Moderado (Skandia)\n2016–{}".format(anio_fin))
plt.xlabel("Fecha"); plt.ylabel("Índice Normalizado (Base 100 en 2016)")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
Path("data/graficas_comparativas").mkdir(parents=True, exist_ok=True)
plt.savefig("data/graficas_comparativas/ipc_vs_skandia_moderado.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Gráfica guardada: data/graficas_comparativas/ipc_vs_skandia_moderado.png")
print("Rangos usados -> IPC:", ipc_b100.index.min().date(), "a", ipc_b100.index.max().date(),
      "| Skandia anual:", sk_b100_anual.index.min().date(), "a", sk_b100_anual.index.max().date())


# ===== IPC vs. TODOS LOS FONDOS DE SKANDIA (gráficas por fondo) =====
# Reutilizamos: ipc_series (anual), base100(s), Path, df_skandia_* ya existentes.

# 1) IPC base 100 (por si no quedó en variable)
ipc_b100 = base100(ipc_series[ipc_series.index >= "2016-01-01"]).rename("IPC (base=100)")

# 2) Helper para nombre de archivo
def _slug(s):
    return (s.lower()
              .replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
              .replace("ñ","n")
              .replace(" ", "_"))

# 3) Fondos Skandia a comparar
fondos_skandia = {
    "Cesantías Corto Plazo"     : df_skandia_cesantias_corto_plazo,
    "Cesantías Largo Plazo"     : df_skandia_cesantias_largo_plazo,
    "Pensiones Moderado"        : df_skandia_pensiones_moderado,
    "Pensiones Conservador"     : df_skandia_pensiones_conservador,
    "Pensiones Mayor Riesgo"    : df_skandia_pensiones_mayor_riesgo,
    "Pensiones Retiro Programado": df_skandia_pensiones_retiro_programado,
    "Pensiones Alternativo"     : df_skandia_pensiones_alternativo,
}

# 4) Graficador por fondo
salidas = []
for nombre, dff in fondos_skandia.items():
    if dff is None or len(dff) == 0:
        print(f"⚠️  Sin datos para: {nombre}")
        continue

    # Serie diaria del fondo
    s = dff.set_index("fecha")["valor_unidad"].sort_index().dropna()
    s = s[s.index >= "2016-01-01"]
    if s.empty:
        print(f"⚠️  Sin datos (>=2016) para: {nombre}")
        continue

    # Anualizar al último dato de cada año (A-DEC = cierre año calendario)
    s_anual = s.resample("A-DEC").last()
    s_b100  = base100(s)           # para ver trayectoria diaria en base 100 (opcional)
    s_b100a = base100(s_anual).rename(f"{nombre} (base=100)")

    # Alinear con IPC (ambos anuales en base 100)
    cmp = pd.concat([ipc_b100, s_b100a], axis=1).sort_index()
    # --- Gráfica ---
    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    # Base 100 (comparables) – EJE IZQUIERDO
    ax.plot(cmp.index, cmp[f"{nombre} (base=100)"], label=f"{nombre} (Base 100)", linewidth=2)
    ax.plot(cmp.index, cmp["IPC (base=100)"], "--", label="IPC Colombia (Base 100)", linewidth=2, marker="o")
    ax.set_ylabel("Índice normalizado (Base 100 en 2016)")
    ax.grid(True, alpha=0.3)

    # Valor de la unidad original anual – EJE DERECHO
    ax2 = ax.twinx()
    ax2.plot(s_anual.index, s_anual.values, ":", linewidth=1.8, color="gray",
             label="Valor unidad (anual, eje der.)")
    ax2.set_ylabel("Valor unidad (anual)")

    # Título, leyendas y guardado
    anio_fin = max(ipc_b100.index.year.max(), s_anual.index.year.max())
    plt.title(f"IPC vs. {nombre} (Skandia) — 2016–{anio_fin}")

    # Combinar leyendas de ambos ejes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left")

    Path("data/graficas_comparativas").mkdir(parents=True, exist_ok=True)
    out = f"data/graficas_comparativas/ipc_vs_skandia_{_slug(nombre)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    salidas.append(out)

print("✓ Gráficas por fondo guardadas:")
for p in salidas:
    print("  -", p)

# ============================================================
# CESANTÍAS CORTO PLAZO (todas las entidades) — Valor anual base 100 (hasta 2024)
# ============================================================

from pathlib import Path

def _valor_anual_base100(df_fondo, nombre):
    """
    Recibe un DataFrame con columnas ['fecha','valor_unidad'] para un fondo.
    Devuelve una Serie: valor anual (cierre A-DEC) hasta 2024.
    """
    if df_fondo is None or len(df_fondo) == 0:
        print(f"⚠️  Sin datos para: {nombre}")
        return None

    s = (df_fondo
         .dropna(subset=['fecha','valor_unidad'])
         .set_index('fecha')['valor_unidad']
         .sort_index())

    # valor de cierre de año calendario (no promedio)
    s_anual = s.resample('A-DEC').last()

    # limitar hasta 2024
    s_anual = s_anual[s_anual.index.year <= 2024]
    if s_anual.empty:
        print(f"⚠️  Serie anual vacía (<=2024) para: {nombre}")
        return None

    # índice = año (int)
    s_anual.index = s_anual.index.year
    s_anual.name = nombre
    return s_anual

# === Crear serie anual del IPC compatible ===
# (ipc_series debe existir: índice datetime en fin de año y valores numéricos)
ipc_anual_df = (
    ipc_series[ipc_series.index.year <= 2024]
    .rename("valor_unidad")
    .reset_index()
    .rename(columns={"index": "fecha"})
)

# === Diccionario solo con DataFrames compatibles (cesantías corto plazo) ===
fondos_corto_dict = {
    "Skandia":      df_skandia_cesantias_corto_plazo,
    "Protección":   df_proteccion_cesantias_corto_plazo,
    "Porvenir":     df_porvenir_cesantias_corto_plazo,
    "Colfondos":    df_colfondos_cesantias_corto_plazo,
    "IPC":          ipc_anual_df,   # ahora con ['fecha','valor_unidad']
}

# 1) Valor anual (cierre) por entidad
series_anuales = {}
for nombre, df_f in fondos_corto_dict.items():
    serie = _valor_anual_base100(df_f, nombre)
    if serie is not None:
        series_anuales[nombre] = serie

# 2) Años comunes (mismo año base y misma malla temporal)
if not series_anuales:
    raise ValueError("No se generaron series anuales para ninguna entidad.")

anios_comunes = set.intersection(*[set(s.index) for s in series_anuales.values()])
anios_comunes = sorted([a for a in anios_comunes if a >= 2016 and a <= 2024])

if len(anios_comunes) < 2:
    raise ValueError(f"No hay años comunes suficientes en 2016–2024. Años comunes: {anios_comunes}")

anio_base = anios_comunes[0]  # típicamente 2016

# 3) Normalizar en base 100 usando el MISMO año base
def _base100_mismo_base(s, base_year, anios):
    s = s.loc[anios].copy()
    base_val = s.loc[base_year]
    return (s / base_val) * 100

series_norm = {nombre: _base100_mismo_base(s, anio_base, anios_comunes)
               for nombre, s in series_anuales.items()}

# 4) Unir a un DataFrame
df_norm = pd.DataFrame(series_norm)
df_norm.index.name = "Año"

# 5) Graficar (IPC destacado)
plt.figure(figsize=(12, 6))

# Dibuja primero las entidades (todas menos IPC)
for nombre in df_norm.columns:
    if nombre != "IPC":
        plt.plot(df_norm.index, df_norm[nombre], marker='o', linewidth=1.8, alpha=0.9, label=nombre)

# Dibuja al final el IPC, para que quede encima y destacado
if "IPC" in df_norm.columns:
    plt.plot(df_norm.index, df_norm["IPC"],
             linestyle='--', linewidth=3.0, marker='s', color='black',
             label="IPC (Base 100)", zorder=10)

plt.title(f"Cesantías Corto Plazo — Valor anual (base 100 en {anio_base})\n(2016–2024)")
plt.xlabel("Año")
plt.ylabel(f"Índice normalizado (base=100 en {anio_base})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

Path("data/graficas_comparativas").mkdir(parents=True, exist_ok=True)
out_png = "data/graficas_comparativas/cesantias_corto_plazo_valor_anual_base100_2016_2024.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

# 6) Exportar tabla con los índices normalizados
Path("data/modelos").mkdir(parents=True, exist_ok=True)
out_csv = "data/modelos/cesantias_corto_plazo_valor_anual_base100_2016_2024.csv"
df_norm.round(2).to_csv(out_csv, encoding="utf-8")

print(f"✓ Gráfica guardada: {out_png}")
print(f"✓ Tabla (base100, años comunes {anios_comunes[0]}–{anios_comunes[-1]}) guardada en: {out_csv}")
print(f"Año base común utilizado: {anio_base}")


# ============================================================
# FASE 4 (Corregida): MODELADO SARIMA EN NIVELES + GRÁFICA AJUSTE vs REAL
# ============================================================

import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# ---------- Utilidades ----------

def _ensure_monthly_close(df, col_fecha='fecha', col_valor='valor_unidad'):
    """
    Toma datos diarios/irregulares y devuelve una Serie mensual (cierre de mes),
    con índice PeriodEnd mensual → DatetimeIndex mensual.
    """
    s = (df[[col_fecha, col_valor]]
         .dropna()
         .sort_values(col_fecha)
         .drop_duplicates(subset=[col_fecha])
         .set_index(col_fecha)[col_valor]
         .astype(float)
         .sort_index())
    # Cierre de mes (ME), luego fijamos frecuencia estricta mensual
    s_m = s.resample('M').last()
    s_m = s_m.asfreq('M')  # asegura malla mensual estricta
    # Completar pequeños huecos con forward fill (no extrapola más allá del inicio)
    s_m = s_m.ffill()
    # Seguridad: Sin valores no positivos (ARIMA puede romper con log), pero aquí no usamos log
    return s_m

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # evita divisiones por 0: ignora términos con y_true==0 para MAPE
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def _grid_search_sarima(y, pdq_grid, seasonal_pdq_grid, trend='c'):
    """
    Búsqueda simple por AIC. y en niveles, frecuencia mensual.
    """
    best = {'aic': np.inf, 'order': None, 'seasonal_order': None, 'model': None}
    for order in pdq_grid:
        for sorder in seasonal_pdq_grid:
            try:
                m = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
                res = m.fit(disp=False)
                if res.aic < best['aic']:
                    best = {'aic': res.aic, 'order': order, 'seasonal_order': sorder, 'model': res}
            except Exception:
                continue
    return best

def _plot_fit_vs_real(y, fit_in, fc, fc_ci, titulo, out_png):
    plt.figure(figsize=(14, 7))
    # Serie real
    plt.plot(y.index, y.values, label='Real (valor de la unidad)', alpha=0.8)
    # Ajuste in-sample (fittedvalues)
    if fit_in is not None:
        plt.plot(fit_in.index, fit_in.values, label='Ajuste in-sample (modelo)', linewidth=2)
    # Pronóstico out-of-sample
    if fc is not None:
        plt.plot(fc.index, fc.values, '--', label='Pronóstico', linewidth=2)
        if fc_ci is not None:
            plt.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], alpha=0.15, label='IC 95%')
    plt.title(titulo)
    plt.xlabel('Fecha'); plt.ylabel('Valor de la unidad')
    plt.grid(True, alpha=0.3); plt.legend()
    Path("data/graficas_modelado").mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()

def entrenar_y_graficar_sarima(df_serie, nombre_serie,
                               meses_test=24, pasos_forecast=12,
                               pdq=((0,0,0),(1,0,0),(0,1,1),(1,1,1),(2,1,2)),
                               spdq=((0,0,0,12),(1,0,0,12),(0,1,1,12),(1,1,1,12))):
    """
    Ajusta SARIMA en niveles (sin diferenciar manualmente), valida en un bloque de test,
    grafica ajuste vs real y guarda métricas.
    """
    # 1) Serie mensual en niveles
    y = _ensure_monthly_close(df_serie, 'fecha', 'valor_unidad')
    if len(y) < (meses_test + 36):
        print(f"⚠️ Serie '{nombre_serie}' es muy corta para validar (len={len(y)}). Se omite.")
        return None

    # Split temporal
    y_train = y.iloc[:-meses_test]
    y_test  = y.iloc[-meses_test:]

    # 2) Búsqueda de hiperparámetros por AIC (en TRAIN)
    pdq_grid = list(pdq)
    seasonal_pdq_grid = list(spdq)
    best = _grid_search_sarima(y_train, pdq_grid, seasonal_pdq_grid, trend='c')
    if best['model'] is None:
        print(f"✗ No se encontró modelo para '{nombre_serie}'.")
        return None

    order, sorder, res = best['order'], best['seasonal_order'], best['model']
    print(f"✓ {nombre_serie}: Mejor SARIMA{order}x{sorder} AIC={best['aic']:.2f}")

    # 3) Predicción dinámica sobre TEST (one-step ahead)
    pred = res.get_prediction(start=y_test.index[0], end=y_test.index[-1], dynamic=False)
    y_pred = pred.predicted_mean
    ci = pred.conf_int()

    # 4) Métricas (en niveles)
    mae  = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mape = _mape(y_test.values, y_pred.values)

    print(f"Resultados '{nombre_serie}' (TEST {y_test.index[0].date()} → {y_test.index[-1].date()}):")
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

    # 5) Pronóstico futuro (pasos_forecast meses)
    fc_res = res.get_forecast(steps=pasos_forecast)
    fc = fc_res.predicted_mean
    fc_ci = fc_res.conf_int()

    # 6) Gráfica ajuste (fitted) + real + test + forecast
    #    Fitted in-sample (en TRAIN): res.fittedvalues
    titulo = (f"{nombre_serie}\n"
              f"SARIMA{order}x{sorder} (AIC={best['aic']:.1f}) | "
              f"Test MAPE={mape:.2f}%")
    out_png = f"data/graficas_modelado/sarima_{nombre_serie.replace(' ','_').lower()}.png"

    # Para que se vea todo junto: real completa, ajuste in-sample (train),
    # y pronóstico solo futuro; la comparación con test ya se imprimió y
    # se puede superponer con get_prediction si deseas.
    _plot_fit_vs_real(y, res.fittedvalues, fc, fc_ci, titulo, out_png)
    print(f"✓ Gráfica guardada: {out_png}")

    # 7) Guardar métricas
    Path("data/modelos").mkdir(parents=True, exist_ok=True)
    pd.Series({
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
        'forecast_steps': pasos_forecast
    }).to_csv(f"data/modelos/sarima_{nombre_serie.replace(' ','_').lower()}_metrics.csv")

    return {
        'nombre_serie': nombre_serie,
        'order': order, 'seasonal_order': sorder,
        'aic': best['aic'], 'mae': mae, 'rmse': rmse, 'mape': mape,
        'fitted': res.fittedvalues, 'forecast': fc, 'forecast_ci': fc_ci,
        'y_train': y_train, 'y_test': y_test, 'y_all': y
    }

# ============================================================
# EJEMPLOS DE EJECUCIÓN (puedes añadir/quitar series)
# ============================================================

resultados_arima = {}

series_a_modelar_nuevas = {
    "Cesantías Corto Plazo - Skandia": df_skandia_cesantias_corto_plazo,
    "Cesantías Corto Plazo - Protección": df_proteccion_cesantias_corto_plazo,
    "Cesantías Corto Plazo - Porvenir": df_porvenir_cesantias_corto_plazo,
    "Cesantías Corto Plazo - Colfondos": df_colfondos_cesantias_corto_plazo,
}

for nombre, dfx in series_a_modelar_nuevas.items():
    if dfx is None or len(dfx) == 0:
        print(f"⚠️ Sin datos para {nombre}")
        continue
    try:
        res = entrenar_y_graficar_sarima(
            dfx, nombre,
            meses_test=24,          # 2 años de test
            pasos_forecast=12,      # 12 meses de pronóstico
            # grids modestos para no demorar (ajusta si quieres más exhaustivo)
            pdq=[(0,0,0),(1,0,0),(0,1,1),(1,1,1),(2,1,2)],
            spdq=[(0,0,0,12),(1,0,0,12),(0,1,1,12),(1,1,1,12)]
        )
        if res:
            resultados_arima[nombre] = res
    except Exception as e:
        print(f"✗ Error modelando {nombre}: {e}")
        
