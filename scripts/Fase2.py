import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Definir la ruta del archivo de entrada

file_path = Path(__file__).parent.parent / "datasets" / "data_filtered.csv"
input_file = pd.read_csv(file_path)

# Directorio de salida: guardar los archivos generados en ../datasets
output_dir = Path(__file__).parent.parent / "datasets"
output_dir.mkdir(parents=True, exist_ok=True)

output_features_file = output_dir / 'data_features_scaled.csv'
output_metadata_file = output_dir / 'data_metadata.csv'

try:
    # Cargar el dataset filtrado
    df = pd.read_csv(file_path)
    print(f"Dataset '{input_file}' cargado. Dimensiones: {df.shape}")
    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    # --- 1. SELECCIÓN DE CARACTERÍSTICAS ---
    
    # Columnas de metadatos (para identificar canciones, no para el modelo)
    # Guardamos 'popularity' y 'year' aquí también por si las usamos luego
    metadata_cols = ['id', 'name', 'artists', 'popularity', 'year', 'release_date']
    
    # Guardar los metadatos para usarlos después de la recomendación
    # Nos aseguramos de dropear duplicados por 'id' si existieran
    metadata_df = df[metadata_cols].drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # Características numéricas que necesitan ser escaladas
    numeric_features_to_scale = [
        'valence', 'acousticness', 'danceability', 'duration_ms', 
        'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness'
    ]
    
    # Características categóricas que necesitan One-Hot Encoding
    # 'key' es categórica (0 a 11)
    categorical_features_to_encode = ['key']
    
    # Características que ya son binarias (0 o 1) y no necesitan escalado
    binary_features = ['mode', 'explicit']

    print(f"\nMetadatos a guardar: {metadata_cols}")
    print(f"Numéricas a escalar: {numeric_features_to_scale}")
    print(f"Categóricas a codificar (OHE): {categorical_features_to_encode}")
    print(f"Binarias (pasar directo): {binary_features}")

    # Asegurarnos de que trabajamos con los mismos datos que los metadatos
    # (por si había duplicados)
    df_features = df[df['id'].isin(metadata_df['id'])].drop_duplicates(subset=['id']).reset_index(drop=True)

    # --- 2. ESCALADO (StandardScaler) ---
    
    # Inicializar el escalador
    scaler = StandardScaler()
    
    # Aplicar el escalador SÓLO a las columnas numéricas
    scaled_numeric_data = scaler.fit_transform(df_features[numeric_features_to_scale])
    
    # Convertir el resultado (numpy array) de nuevo a un DataFrame
    scaled_numeric_df = pd.DataFrame(scaled_numeric_data, 
                                     columns=numeric_features_to_scale, 
                                     index=df_features.index) # Mantener el índice original
    
    print(f"\nValores numéricos escalados (primeras 5 filas):")
    print(scaled_numeric_df.head())

    # --- 3. CODIFICACIÓN (One-Hot Encoding) ---
    
    # Aplicar One-Hot Encoding a 'key'
    # prefix='key' crea columnas como key_0, key_1, ...
    encoded_categorical_df = pd.get_dummies(df_features[categorical_features_to_encode].astype(str), 
                                            prefix='key', 
                                            drop_first=False) # No dropear la primera para KNN/Clustering
    
    print(f"\nCaracterísticas categóricas codificadas (primeras 5 filas):")
    print(encoded_categorical_df.head())

    # --- 4. COMBINACIÓN FINAL ---
    
    # Unir todas las características procesadas en un solo DataFrame
    # scaled_numeric_df | encoded_categorical_df | binary_features
    
    features_df = pd.concat([
        scaled_numeric_df, 
        encoded_categorical_df, 
        df_features[binary_features].reset_index(drop=True) # reset_index para alinear
    ], axis=1)

    print(f"\nDataFrame final de características (primeras 5 filas):")
    print(features_df.head())
    print(f"\nDimensiones finales del DataFrame de características: {features_df.shape}")

    # --- 5. GUARDAR ARCHIVOS ---
    
    # Guardar el DataFrame de características escaladas y codificadas
    features_df.to_csv(output_features_file, index=False)
    print(f"\n✅ Archivo de características guardado en: {output_features_file}")
    
    # Guardar el DataFrame de metadatos
    metadata_df.to_csv(output_metadata_file, index=False)
    print(f"✅ Archivo de metadatos guardado en: {output_metadata_file}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{input_file}'. Asegúrate de que está en el directorio correcto.")
except KeyError as e:
    print(f"Error de columna (KeyError): No se encontró la columna {e}. Verifica que 'data_filtered.csv' tiene todas las columnas esperadas.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")