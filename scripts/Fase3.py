import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Leer ahora el CSV filtrado desde la carpeta datasets (ruta absoluta relativa al repo)
file_path = Path(__file__).parent.parent / "datasets" / "data_filtered.csv"
try:
    input_file = pd.read_csv(file_path)
    print(f"Archivo de entrada cargado en import: {file_path}")
except FileNotFoundError:
    input_file = None
    print(f"Warning: No se encontró el archivo de entrada en: {file_path}")


# Cargar también metadata y features en import (como solicitaste)
metadata_path = Path(__file__).parent.parent / "datasets" / "data_metadata.csv"
features_path = Path(__file__).parent.parent / "datasets" / "data_features_scaled.csv"
try:
    metadata_file = pd.read_csv(metadata_path)
    print(f"Archivo de metadatos cargado en import: {metadata_path}")
except FileNotFoundError:
    metadata_file = None
    print(f"Warning: No se encontró el archivo de metadatos en: {metadata_path}")

try:
    features_file = pd.read_csv(features_path)
    print(f"Archivo de features cargado en import: {features_path}")
except FileNotFoundError:
    features_file = None
    print(f"Warning: No se encontró el archivo de features en: {features_path}")


# Definir la ruta al directorio de datasets (relativa al repo/script)
DATA_DIR = Path(__file__).parent.parent / 'datasets'
METADATA_FILE = DATA_DIR / 'data_metadata.csv'
FEATURES_FILE = DATA_DIR / 'data_features_scaled.csv'

def load_data():
    """Carga los dataframes de metadatos y features."""
    try:
        # Si ya cargamos los DataFrames en import, úsalos directamente
        if 'metadata_file' in globals() and metadata_file is not None and \
           'features_file' in globals() and features_file is not None:
            metadata_df = metadata_file
            features_df = features_file
        else:
            metadata_df = pd.read_csv(METADATA_FILE)
            features_df = pd.read_csv(FEATURES_FILE)
        
        print(f"Metadatos cargados: {metadata_df.shape}")
        print(f"Features cargadas: {features_df.shape}")
        
        if len(metadata_df) != len(features_df):
            print("¡Advertencia! El número de filas no coincide entre metadatos y features.")
        
        return metadata_df, features_df
    
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}.")
        print("Asegúrate de haber ejecutado el script de escalado (Fase 2) primero.")
        return None, None

# --- 2. ENTRENAMIENTO DEL MODELO KNN ---

def train_knn_model(features_df, k=10, metric='euclidean'):
    """
    Entrena el modelo NearestNeighbors con las features dadas.
    'k' es el número de vecinos a buscar (incluyendo la propia canción).
    """
    # Usamos k+1 porque el vecino más cercano (distancia 0) será la misma canción
    model = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
    
    print(f"\nEntrenando modelo KNN con k={k} y métrica='{metric}'...")
    model.fit(features_df)
    print("✅ Modelo entrenado.")
    
    return model

# --- 3. FUNCIONES DE RECOMENDACIÓN ---

def find_song_index(song_name, artist_name, metadata_df):
    """
    Busca el índice (fila) de una canción por nombre y artista.
    Devuelve None si no la encuentra.
    """
    # Búsqueda (simplificada, sensible a mayúsculas)
    # Para un proyecto más avanzado, podrías usar fuzzy matching
    
    # Primero buscamos por nombre
    possible_matches = metadata_df[metadata_df['name'].str.contains(song_name, case=False)]
    
    if possible_matches.empty:
        return None
    
    # Si hay varios, filtramos por artista
    if artist_name:
        final_match = possible_matches[possible_matches['artists'].str.contains(artist_name, case=False)]
        if not final_match.empty:
            return final_match.index[0] # Devolvemos el índice de la primera coincidencia
    
    # Si no hay filtro de artista o no coincide, devolvemos el primero
    return possible_matches.index[0]

def recommend_songs(song_index, model, features_df, metadata_df, k=5):
    """
    Obtiene recomendaciones para una canción dado su índice.
    """
    if song_index is None or song_index >= len(features_df):
        return pd.DataFrame() # Devuelve un DataFrame vacío si el índice es inválido

    # Obtener el vector de features de la canción de entrada
    song_features = features_df.iloc[[song_index]]
    
    # Encontrar los k+1 vecinos más cercanos
    distances, indices = model.kneighbors(song_features, n_neighbors=k+1)
    
    # Los 'indices' son las filas en features_df (y metadata_df)
    # El primer índice (indices[0][0]) es la propia canción, así que lo saltamos
    recommended_indices = indices[0][1:]
    
    # Obtener los metadatos de las canciones recomendadas
    recommended_songs = metadata_df.iloc[recommended_indices]
    
    # Añadir la distancia (qué tan similar es)
    recommended_songs['distance'] = distances[0][1:]
    
    return recommended_songs[['name', 'artists', 'year', 'popularity', 'distance']]

# --- 4. EJEMPLO DE USO ---

if __name__ == "__main__":
    
    # Cargar los datos
    metadata, features = load_data()
    
    if metadata is not None and features is not None:
        
        # Entrenar el modelo
        # Buscamos 6 vecinos (5 recomendaciones + la original)
        knn_model = train_knn_model(features, k=6, metric='euclidean')
        
        # --- ¡PRUEBA AQUÍ CON TUS CANCIONES! ---
        
        # Ejemplo 1: Buscar "No Scrubs" de "TLC" (estaba al inicio de tu CSV)
        test_song_name = "No Scrubs"
        test_artist_name = "TLC"
        
        print("\n" + "="*50)
        print(f"Buscando recomendaciones para: '{test_song_name}' por '{test_artist_name}'")
        
        # 1. Encontrar el índice de la canción
        song_idx = find_song_index(test_song_name, test_artist_name, metadata)
        
        if song_idx is not None:
            print(f"Canción encontrada en el índice: {song_idx}")
            original_song = metadata.iloc[song_idx]
            print(f"-> {original_song['name']} - {original_song['artists']} ({original_song['year']})")
            
            # 2. Obtener recomendaciones
            recommendations = recommend_songs(song_idx, knn_model, features, metadata, k=5)
            
            print("\n--- Recomendaciones Encontradas ---")
            print(recommendations)
            
        else:
            print(f"No se pudo encontrar la canción '{test_song_name}'. Intenta con otro nombre.")
        
        # Ejemplo 2: Probar con otra canción
        test_song_name_2 = "All The Small Things"
        test_artist_name_2 = "blink-182"
        
        print("\n" + "="*50)
        print(f"Buscando recomendaciones para: '{test_song_name_2}' por '{test_artist_name_2}'")
        
        song_idx_2 = find_song_index(test_song_name_2, test_artist_name_2, metadata)
        
        if song_idx_2 is not None:
            print(f"Canción encontrada en el índice: {song_idx_2}")
            original_song_2 = metadata.iloc[song_idx_2]
            print(f"-> {original_song_2['name']} - {original_song_2['artists']} ({original_song_2['year']})")
            
            recommendations_2 = recommend_songs(song_idx_2, knn_model, features, metadata, k=5)
            
            print("\n--- Recomendaciones Encontradas ---")
            print(recommendations_2)
            
        else:
            print(f"No se pudo encontrar la canción '{test_song_name_2}'.")