import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pathlib import Path

import matplotlib.pyplot as plt


# Load iris dataset
# ==================== 1. CARGA DE DATOS ====================
file_path = Path(__file__).parent.parent / "datasets" / "data.csv"
df = pd.read_csv(file_path)

print("="*50)
print("ANÁLISIS EXPLORATORIO INICIAL")
print("="*50)
print(f"\nDimensiones del dataset: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nInformación del dataset:\n{df.info()}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")

# ==================== 2. FILTRADO DE DATOS ====================
print("\n" + "="*50)
print("FILTRADO DE DATOS")
print("="*50)

# Filtrar canciones anteriores a 1979
print(f"\nFilas antes del filtrado por año: {df.shape[0]}")
df = df[df['year'] >= 1999]
print(f"Filas después del filtrado (year >= 1999): {df.shape[0]}")

# Filtrar loudness con valor absoluto > 8 dígitos
print(f"\nFilas antes del filtrado por loudness: {df.shape[0]}")
# Contar dígitos del valor absoluto (sin punto decimal ni signo)
df['loudness_digits'] = df['loudness'].abs().astype(str).str.replace('.', '').str.len()
df = df[df['loudness_digits'] <= 8]
df = df.drop('loudness_digits', axis=1)
print(f"Filas después del filtrado (loudness con <= 8 dígitos): {df.shape[0]}")


# Eliminar columna tempo
print(f"\nColumnas antes de eliminar tempo: {df.shape[1]}")
df = df.drop('tempo', axis=1)
print(f"Columnas después de eliminar tempo: {df.shape[1]}")


print(f"\n✅ Dataset filtrado final: {df.shape}")
print(f"\nPrimeras filas después del filtrado:\n{df.head()}")
print(f"\nEstadísticas de loudness después del filtrado:\n{df['loudness'].describe()}")


# ==================== 3. GUARDAR DATASET FILTRADO ====================
output_path = Path(__file__).parent.parent / "datasets" / "data_filtered.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Dataset filtrado guardado en: {output_path}")
print(f"Total de filas guardadas: {df.shape[0]}")