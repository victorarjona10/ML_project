import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================

# Definimos la ruta base usando __file__ para ser robustos independientemente de dónde se ejecute el script
# __file__ es este archivo. .parent es la carpeta donde está. .parent.parent sube un nivel más.
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

print(f"📂 Buscando datasets en: {DATASETS_DIR.resolve()}")

# ==========================================
# 2. LÓGICA DE MACHINE LEARNING
# ==========================================

# Variables globales
model_knn = None
df_metadata = None
df_features = None

def load_and_train():
    """
    Carga los datos y entrena el modelo al iniciar la API.
    """
    global model_knn, df_metadata, df_features
    
    print("⏳ Cargando datasets...")
    try:
        # Verificamos si los archivos existen antes de intentar leerlos
        if not METADATA_FILE.exists():
            print(f"❌ Error: No se encontró el archivo de metadatos en: {METADATA_FILE}")
            return False
            
        if not FEATURES_FILE.exists():
            print(f"❌ Error: No se encontró el archivo de features en: {FEATURES_FILE}")
            return False

        # Cargar archivos
        df_metadata = pd.read_csv(METADATA_FILE)
        df_features = pd.read_csv(FEATURES_FILE)
        
        print(f"✅ Datos cargados correctamente.")
        print(f"   - Metadatos: {len(df_metadata)} canciones")
        print(f"   - Features: {df_features.shape} dimensiones")

        # Entrenar el modelo KNN
        print("⏳ Entrenando modelo KNN...")
        model_knn = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='auto')
        model_knn.fit(df_features)
        print("✅ Modelo KNN entrenado y listo para recibir peticiones.")
        return True
        
    except Exception as e:
        print(f"❌ Error crítico al iniciar: {e}")
        return False

def find_song_index(song_name, artist_name):
    """Busca el índice de una canción en el dataframe de metadatos."""
    if df_metadata is None: return None

    # Búsqueda por nombre (insensible a mayúsculas)
    matches = df_metadata[df_metadata['name'].str.contains(song_name, case=False, na=False)]
    
    if matches.empty:
        return None
    
    # Si nos dan artista, filtramos también
    if artist_name:
        artist_matches = matches[matches['artists'].str.contains(artist_name, case=False, na=False)]
        if not artist_matches.empty:
            matches = artist_matches
    
    # Devolvemos el índice de la primera coincidencia
    return matches.index[0]

def get_recommendations_logic(song_index, k=5):
    """Obtiene las k canciones más cercanas."""
    # Obtener vector de la canción
    song_vector = df_features.iloc[[song_index]]
    
    # Buscar vecinos
    distances, indices = model_knn.kneighbors(song_vector, n_neighbors=k+1)
    
    # Ignorar el primero (es la misma canción)
    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]
    
    # Obtener datos de las recomendadas
    recommendations = df_metadata.iloc[rec_indices].copy()
    recommendations['similarity_distance'] = rec_distances
    
    # Seleccionar solo columnas útiles para devolver
    cols_to_return = ['name', 'artists', 'year', 'popularity', 'similarity_distance']
    return recommendations[cols_to_return].to_dict(orient='records')

# ==========================================
# 3. DEFINICIÓN DE LA API (FastAPI)
# ==========================================

app = FastAPI(
    title="API Recomendador de Música",
    description="API KNN para recomendar canciones.",
    version="1.0"
)

class SongRequest(BaseModel):
    song_name: str
    artist_name: str = "" 

# --- Evento de inicio ---
@app.on_event("startup")
def startup_event():
    success = load_and_train()
    if not success:
        print("⚠️ ADVERTENCIA: La API arrancó pero el modelo NO se cargó correctamente.")

# --- Endpoints ---
@app.get("/")
def home():
    return {"message": "API funcionando. Ve a /docs para probar el recomendador."}

@app.post("/recommend")
def recommend(request: SongRequest):
    if model_knn is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Revisa los logs del servidor.")

    print(f"🔍 Petición recibida: '{request.song_name}' - '{request.artist_name}'")

    idx = find_song_index(request.song_name, request.artist_name)
    
    if idx is None:
        raise HTTPException(status_code=404, detail=f"No se encontró la canción '{request.song_name}'")

    found_song = df_metadata.iloc[idx]
    
    try:
        recs = get_recommendations_logic(idx, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    return {
        "song_found": {
            "name": found_song['name'],
            "artist": found_song['artists'],
            "year": int(found_song['year'])
        },
        "recommendations": recs
    }

# ==========================================
# 4. EJECUCIÓN DEL SERVIDOR
# ==========================================

if __name__ == "__main__":
    print("🚀 Iniciando servidor Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)