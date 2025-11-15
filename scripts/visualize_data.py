import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. CARGA DE DATOS ====================
print("="*60)
print("VISUALIZACIÓN DE DATOS - ANÁLISIS EXPLORATORIO")
print("="*60)

file_path = Path(__file__).parent.parent / "datasets" / "data_filtered.csv"
df = pd.read_csv(file_path)

print(f"\n📊 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nColumnas disponibles:\n{df.columns.tolist()}")
print(f"\nPrimeras filas:\n{df.head()}")

# Identificar columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n📈 Columnas numéricas: {numeric_cols}")

# ==================== 2. VISUALIZACIONES ====================

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 12))

# --- 2.1. Distribución de año ---
plt.subplot(3, 3, 1)
plt.hist(df['year'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Año', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Canciones por Año (≥1999)', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# --- 2.2. Distribución de loudness ---
plt.subplot(3, 3, 2)
plt.hist(df['loudness'], bins=40, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Loudness (dB)', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Loudness', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# --- 2.3. Boxplot de loudness ---
plt.subplot(3, 3, 3)
plt.boxplot(df['loudness'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Loudness (dB)', fontsize=10)
plt.title('Boxplot de Loudness', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# --- 2.4. Canciones por año (barras) ---
plt.subplot(3, 3, 4)
year_counts = df['year'].value_counts().sort_index()
plt.bar(year_counts.index, year_counts.values, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Año', fontsize=10)
plt.ylabel('Número de Canciones', fontsize=10)
plt.title('Cantidad de Canciones por Año', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# --- 2.5. Evolución de loudness por año ---
plt.subplot(3, 3, 5)
loudness_by_year = df.groupby('year')['loudness'].mean()
plt.plot(loudness_by_year.index, loudness_by_year.values, marker='o', linewidth=2, markersize=4)
plt.xlabel('Año', fontsize=10)
plt.ylabel('Loudness Promedio (dB)', fontsize=10)
plt.title('Evolución de Loudness a lo Largo del Tiempo', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# --- 2.6. Matriz de correlación ---
if len(numeric_cols) > 2:
    plt.subplot(3, 3, 6)
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación', fontsize=12, fontweight='bold')

# --- 2.7. Distribución de otras variables numéricas ---
if len(numeric_cols) > 2:
    # Excluir year y loudness para ver otras variables
    other_numeric = [col for col in numeric_cols if col not in ['year', 'loudness']]
    
    if len(other_numeric) >= 1:
        plt.subplot(3, 3, 7)
        plt.hist(df[other_numeric[0]], bins=30, edgecolor='black', alpha=0.7, color='purple')
        plt.xlabel(other_numeric[0], fontsize=10)
        plt.ylabel('Frecuencia', fontsize=10)
        plt.title(f'Distribución de {other_numeric[0]}', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    if len(other_numeric) >= 2:
        plt.subplot(3, 3, 8)
        plt.hist(df[other_numeric[1]], bins=30, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel(other_numeric[1], fontsize=10)
        plt.ylabel('Frecuencia', fontsize=10)
        plt.title(f'Distribución de {other_numeric[1]}', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    if len(other_numeric) >= 2:
        plt.subplot(3, 3, 9)
        plt.scatter(df[other_numeric[0]], df[other_numeric[1]], alpha=0.5, s=10)
        plt.xlabel(other_numeric[0], fontsize=10)
        plt.ylabel(other_numeric[1], fontsize=10)
        plt.title(f'{other_numeric[0]} vs {other_numeric[1]}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "visualizations" / "exploratory_analysis.png", 
            dpi=300, bbox_inches='tight')
print("\n💾 Gráficas guardadas en: visualizations/exploratory_analysis.png")
plt.show()

# ==================== 3. ESTADÍSTICAS ADICIONALES ====================
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS DETALLADAS")
print("="*60)

print(f"\n📊 Estadísticas de Loudness:")
print(df['loudness'].describe())

print(f"\n📊 Estadísticas por Año:")
print(f"  - Año mínimo: {df['year'].min()}")
print(f"  - Año máximo: {df['year'].max()}")
print(f"  - Rango: {df['year'].max() - df['year'].min()} años")

print(f"\n📊 Top 5 años con más canciones:")
print(df['year'].value_counts().head())

print("\n✅ Análisis completado!")
