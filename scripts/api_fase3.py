import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================

# Ruta robusta usando __file__
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

# ==========================================
# 2. LÓGICA DE MACHINE LEARNING
# ==========================================

model_knn = None
df_metadata = None
df_features = None

def load_and_train():
    global model_knn, df_metadata, df_features
    print("⏳ Cargando datasets...")
    try:
        if not METADATA_FILE.exists() or not FEATURES_FILE.exists():
            print(f"❌ Error: No se encuentran archivos en {DATASETS_DIR}")
            return False

        df_metadata = pd.read_csv(METADATA_FILE)
        df_features = pd.read_csv(FEATURES_FILE)
        
        # Limpieza básica para asegurar que las búsquedas de texto funcionen bien
        df_metadata['name'] = df_metadata['name'].astype(str)
        df_metadata['artists'] = df_metadata['artists'].astype(str)

        print(f"✅ Datos cargados: {len(df_metadata)} canciones.")

        print("⏳ Entrenando modelo KNN...")
        model_knn = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='auto')
        model_knn.fit(df_features)
        print("✅ Modelo KNN listo.")
        return True
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        return False

def find_song_index(song_name, artist_name):
    """
    Busca el índice de una canción asegurando que coincida Artista y Canción.
    """
    if df_metadata is None: return None

    # 1. Primero buscamos coincidencias por Nombre de canción
    # case=False: ignora mayúsculas/minúsculas
    # na=False: ignora valores nulos
    song_matches = df_metadata[df_metadata['name'].str.contains(song_name, case=False, na=False)]
    
    if song_matches.empty:
        return None
    
    # 2. Si el usuario especificó artista, filtramos ESTRICTAMENTE esos resultados
    if artist_name:
        # Buscamos dentro de las canciones que ya coincidieron en nombre
        artist_matches = song_matches[song_matches['artists'].str.contains(artist_name, case=False, na=False)]
        
        if not artist_matches.empty:
            # ¡ÉXITO! Encontramos Canción + Artista
            # Devolvemos el primer índice de esta coincidencia exacta
            return artist_matches.index[0]
        else:
            # CASO CLAVE: Existe la canción, pero NO con ese artista.
            # Devolvemos None para no dar una canción equivocada.
            print(f"DEBUG: Se encontró la canción '{song_name}' pero no del artista '{artist_name}'.")
            return None
    
    # 3. Si el usuario NO puso artista, devolvemos la primera coincidencia del nombre
    return song_matches.index[0]

def get_recommendations_logic(song_index, k=5):
    song_vector = df_features.iloc[[song_index]]
    distances, indices = model_knn.kneighbors(song_vector, n_neighbors=k+1)
    
    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]
    
    recommendations = df_metadata.iloc[rec_indices].copy()
    recommendations['similarity_distance'] = rec_distances
    
    cols_to_return = ['name', 'artists', 'year', 'popularity', 'similarity_distance']
    return recommendations[cols_to_return].to_dict(orient='records')

# ==========================================
# 3. DEFINICIÓN DE LA API
# ==========================================

app = FastAPI(
    title="API Recomendador Musical",
    description="API para recomendar canciones usando KNN.",
    version="1.1"
)

class SongRequest(BaseModel):
    song_name: str
    artist_name: str = "" 

@app.on_event("startup")
def startup_event():
    load_and_train()

@app.get("/")
def home():
    return {"message": "API Recomendador v1.1 Activa"}

@app.post("/recommend")
def recommend(request: SongRequest):
    if model_knn is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")

    print(f"🔍 Buscando: '{request.song_name}' de '{request.artist_name}'")

    idx = find_song_index(request.song_name, request.artist_name)
    
    if idx is None:
        # Mensaje de error más descriptivo
        detail_msg = f"No se encontró la canción '{request.song_name}'"
        if request.artist_name:
            detail_msg += f" del artista '{request.artist_name}'"
        
        raise HTTPException(status_code=404, detail=detail_msg)

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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)